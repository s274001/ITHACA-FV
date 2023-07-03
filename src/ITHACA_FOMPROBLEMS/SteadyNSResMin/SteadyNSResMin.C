/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

  License
  This file is part of ITHACA-FV

  ITHACA-FV is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ITHACA-FV is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/


/// \file
/// Source file of the steadyNS class.

#include "SteadyNSResMin.H"
#include "viscosityModel.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
// Constructor
SteadyNSResMin::SteadyNSResMin() {}
SteadyNSResMin::SteadyNSResMin(int argc, char* argv[])
{
    _args = autoPtr<argList>
            (
                new argList(argc, argv)
            );

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );
    simpleControl& simple = _simple();
#include "createFields.H"
#include "createFvOptions.H"
    turbulence->validate();
    ITHACAdict = new IOdictionary
    (
        IOobject
        (
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    tolerance = ITHACAdict->lookupOrDefault<scalar>("tolerance", 1e-5);
    maxIter = ITHACAdict->lookupOrDefault<scalar>("maxIter", 1000);
    bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "lift");
    M_Assert(bcMethod == "lift" || bcMethod == "penalty" || bcMethod == "none",
             "The BC method must be set to lift or penalty or none in ITHACAdict");
    fluxMethod = ITHACAdict->lookupOrDefault<word>("fluxMethod", "inconsistent");
    M_Assert(fluxMethod == "inconsistent" || bcMethod == "consistent",
             "The flux method must be set to inconsistent or consistent in ITHACAdict");
    para = ITHACAparameters::getInstance(mesh, runTime);
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
    supex = ITHACAutilities::check_sup();
}

void SteadyNSResMin::computeResidual(volVectorField& U, volScalarField& p,
                                     Eigen::MatrixXd& resU, Eigen::MatrixXd& resP)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    surfaceScalarField& phi = _phi();
    fv::options& fvOptions = _fvOptions();
    simpleControl& simple = _simple();
    IOMRFZoneList& MRF = _MRF();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    this->turbulence->validate();
    MRF.correctBoundaryVelocity(U);
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
        + MRF.DDt(U) //non c'è
        + turbulence->divDevReff(U)
        ==
        fvOptions(U)    // non c'è
        - fvc::grad(p)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    Foam::vectorField tres = UEqn.residual();
    resU = Foam2Eigen::field2Eigen(tres);
    tmp<fvVectorMatrix> tUEqn2(UEqn == fvc::grad(p));
    fvVectorMatrix& UEqn2 = tUEqn2.ref();
    UEqn2.relax();
    dimensionedScalar small("small", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-18);
    volScalarField rAU(1.0 / UEqn2.A());
    volVectorField HbyA(constrainHbyA(rAU * UEqn2.H(), U, p));
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
    MRF.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U, p);
    tmp<volScalarField> rAtU(rAU);

    if (simple.consistent())
    {
        rAtU = 1.0 / (1.0 / rAU - UEqn2.H1());
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p) * mesh.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p);
    }

    constrainPressure(p, U, phiHbyA, rAtU(), MRF);
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
    );
    Foam::scalarField tpres = pEqn.residual();
    resP = Foam2Eigen::field2Eigen(tpres);
}
