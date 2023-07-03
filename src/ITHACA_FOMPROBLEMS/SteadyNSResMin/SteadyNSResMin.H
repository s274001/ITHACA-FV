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
Class
    steadyNS
Description
    Reduction class for a Stationary Navier-Stokes problem.
SourceFiles
    steadyNS.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the steadyNSResMin class.
/// \dir
/// Directory containing the header and source files for the steadyNSResMin class.

#ifndef SteadyNSResMin_H
#define SteadyNSResMin_H
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "pisoControl.H"
#include "fvOptions.H"
#include "steadyNS.H"
#include "ITHACAstream.H"
#include "ITHACAparameters.H"
#if OPENFOAM >= 1812
#include "ITHACAforces18.H"
#else
#include "ITHACAforces.H"
#endif
#include "volFields.H"
#include <iostream>
#include "IPstream.H"
#include "OPstream.H"
#include "Modes.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

/*---------------------------------------------------------------------------*\
                        Class SteadyNS Declaration
\*---------------------------------------------------------------------------*/

/// Implementation of a parametrized full order <b> steady NS problem </b> and preparation of the the reduced matrices for the online solve.
/** In this class are implemented the methods for the offline solve of a steady NS problem
and the for the generation of the reduced matrices for subsequent online solve, this class is a son
of the reduction problem class */
class SteadyNSResMin: public steadyNS
{
    public:
        // Constructors
        /// Null constructor
        SteadyNSResMin();

        /// Construct with argc and argv
        SteadyNSResMin(int argc, char* argv[]);
        ~SteadyNSResMin() {};

        void computeResidual(volVectorField& U, volScalarField& p,
                             Eigen::MatrixXd& resU, Eigen::MatrixXd& resP);

        autoPtr<fvMeshSubset> submesh;
};

#endif










