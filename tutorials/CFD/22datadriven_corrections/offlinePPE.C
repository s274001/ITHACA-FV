/*---------------------------------------------------------------------------*\
Copyright (C) 2017 by the ITHACA-FV authors

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

Description
    Example of NS-Stokes Reduction Problem

\*---------------------------------------------------------------------------*/
#include "steadyNS.H"
#include "UnsteadyNSTurb.H"
#include <math.h>

#include "ITHACAutilities.H"
#include "reductionProblem.H"

#include "ReducedSteadyNS.H"
#include "ReducedUnsteadyNSTurb.H"
#include "ReducedUnsteadyNS.H"

#include <Eigen/Dense>
#include <Eigen/Core>
#include "forces.H"
#include "IOmanip.H"
#include "forces.H"
#include "forceCoeffs.H"
#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "Foam2Eigen.H"
#include <chrono>
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "EigenFunctions.H"
#include <chrono>
#include <Eigen/SVD>
#include <Eigen/SparseLU>

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>


class tutorial22ppe : public UnsteadyNSTurb
{
    public:
        explicit tutorial22ppe(int argc, char *argv[])
            :
            UnsteadyNSTurb(argc, argv),
            U(_U()),
            p(_p()),
            nut(_nut())
        {}

        // Relevant Fields
        volVectorField& U;
        volScalarField& p;
        volScalarField& nut;

        void offlineSolve()
        {
            Vector<double> inl(0, 0, 0);
            List<scalar> mu_now(1);
            mu_now[0] =  1e-05;

            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(nutFields, nut, "./ITHACAoutput/Offline/");
            }

            else
            {
                truthSolve(mu_now);
            }
        }
        Eigen::MatrixXd vectorTensorMult(Eigen::VectorXd g, Eigen::Tensor<double, 3> c ,
                                         Eigen::VectorXd a)
        {
            int prodDim = c.dimension(0);
            Eigen::MatrixXd prod;
            prod.resize(prodDim, 1);

            for (int i = 0; i < prodDim; i++)
            {
                prod(i, 0) = g.transpose() *
                             Eigen::SliceFromTensor(c, 0, i) * a;
            }

            return prod;
        }
        
        void projectPPE(fileName folder, label NU, label NP, label NSUP,
                                        label Nnut)
        {
            NUmodes = NU;
            NPmodes = NP;
            NSUPmodes = 0;
            nNutModes = Nnut;
            L_U_SUPmodes.resize(0);

            if (liftfield.size() != 0)
            {
                for (label k = 0; k < liftfield.size(); k++)
                {
                    L_U_SUPmodes.append(tmp<volVectorField>(liftfield[k]));
                }
            }

            if (NUmodes != 0)
            {
                for (label k = 0; k < NUmodes; k++)
                {
                    L_U_SUPmodes.append(tmp<volVectorField>(Umodes[k]));
                }
            }

            if (ITHACAutilities::check_folder("./ITHACAoutput/Matrices/"))
            {
                word B_str = "B_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + B_str))
                {
                    ITHACAstream::ReadDenseMatrix(B_matrix, "./ITHACAoutput/Matrices/", B_str);
                }
                else
                {
                    B_matrix = diffusive_term(NUmodes, NPmodes, NSUPmodes);
                }

                word btStr = "bt_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + btStr))
                {
                    ITHACAstream::ReadDenseMatrix(btMatrix, "./ITHACAoutput/Matrices/", btStr);
                }
                else
                {
                    btMatrix = btTurbulence(NUmodes, NSUPmodes);
                }

                word K_str = "K_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes) + "_" + name(NPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + K_str))
                {
                    ITHACAstream::ReadDenseMatrix(K_matrix, "./ITHACAoutput/Matrices/", K_str);
                }
                else
                {
                    K_matrix = pressure_gradient_term(NUmodes, NPmodes, NSUPmodes);
                }

                word M_str = "M_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + M_str))
                {
                    ITHACAstream::ReadDenseMatrix(M_matrix, "./ITHACAoutput/Matrices/", M_str);
                }
                else
                {
                    M_matrix = mass_term(NUmodes, NPmodes, NSUPmodes);
                }

                word D_str = "D_" + name(NPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + D_str))
                {
                    ITHACAstream::ReadDenseMatrix(D_matrix, "./ITHACAoutput/Matrices/", D_str);
                }
                else
                {
                    D_matrix = laplacian_pressure(NPmodes);
                }

                word bc1_str = "BC1_" + name(liftfield.size()) + "_" + name(
                                   NUmodes) + "_" + name(NPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + bc1_str))
                {
                    ITHACAstream::ReadDenseMatrix(BC1_matrix, "./ITHACAoutput/Matrices/", bc1_str);
                }
                else
                {
                    BC1_matrix = pressure_BC1(NUmodes, NPmodes);
                }

                word bc2_str = "BC2_" + name(liftfield.size()) + "_" + name(
                                   NUmodes) + "_" + name(
                                   NSUPmodes) + "_" + name(NPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + bc2_str))
                {
                    ITHACAstream::ReadDenseTensor(bc2Tensor, "./ITHACAoutput/Matrices/", bc2_str);
                }
                else
                {
                    bc2Tensor = pressureBC2(NUmodes, NPmodes);
                }

                word bc3_str = "BC3_" + name(liftfield.size()) + "_" + name(
                                   NUmodes) + "_" + name(NPmodes);

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + bc3_str))
                {
                    ITHACAstream::ReadDenseMatrix(BC3_matrix, "./ITHACAoutput/Matrices/", bc3_str);
                }
                else
                {
                    BC3_matrix = pressure_BC3(NUmodes, NPmodes);
                }

                word C_str = "C_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + C_str))
                {
                    ITHACAstream::ReadDenseTensor(C_tensor, "./ITHACAoutput/Matrices/", C_str);
                }
                else
                {
                    C_tensor = convective_term_tens(NUmodes, NPmodes, NSUPmodes);
                }

                word ct1Str = "ct1_" + name(liftfield.size()) + "_" + name(
                                  NUmodes) + "_" + name(
                                  NSUPmodes) + "_" + name(nNutModes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct1Str))
                {
                    ITHACAstream::ReadDenseTensor(ct1Tensor, "./ITHACAoutput/Matrices/", ct1Str);
                }
                else
                {
                    ct1Tensor = turbulenceTensor1(NUmodes, NSUPmodes, nNutModes);
                }

                word ct2Str = "ct2_" + name(liftfield.size()) + "_" + name(
                                  NUmodes) + "_" + name(
                                  NSUPmodes) + "_" + name(nNutModes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct2Str))
                {
                    ITHACAstream::ReadDenseTensor(ct2Tensor, "./ITHACAoutput/Matrices/", ct2Str);
                }    
                else
                {
                    ct2Tensor = turbulenceTensor2(NUmodes, NSUPmodes, nNutModes);
                }

                word G_str = "G_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                 NSUPmodes) + "_" + name(NPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + G_str))
                {
                    ITHACAstream::ReadDenseTensor(gTensor, "./ITHACAoutput/Matrices/", G_str);
                }
                else
                {
                    gTensor = divMomentum(NUmodes, NPmodes);
                }

                word ct1AveStr = "ct1Ave_" + name(liftfield.size()) + "_" + name(
                                     NUmodes) + "_" + name(
                                     NSUPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct1AveStr))
                {
                    ITHACAstream::ReadDenseTensor(ct1AveTensor, "./ITHACAoutput/Matrices/",
                                          ct1AveStr);
                }
                else
                {
                    ct1AveTensor = turbulenceAveTensor1(NUmodes, NSUPmodes);
                }

                word ct2AveStr = "ct2Ave_" + name(liftfield.size()) + "_" + name(
                                     NUmodes) + "_" + name(
                                     NSUPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct2AveStr))
                {
                    ITHACAstream::ReadDenseTensor(ct2AveTensor, "./ITHACAoutput/Matrices/",
                                                  ct2AveStr);
                }
                else
                {
                    ct2AveTensor = turbulenceAveTensor2(NUmodes, NSUPmodes);
                }

                word ct1PPEStr = "ct1PPE_" + name(liftfield.size()) + "_" + name(
                                     NUmodes) + "_" + name(
                                     NSUPmodes) + "_" + name(NPmodes) + "_" + name(nNutModes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct1PPEStr))
                {
                    ITHACAstream::ReadDenseTensor(ct1PPETensor, "./ITHACAoutput/Matrices/",
                                                  ct1PPEStr);
                }
                else
                {
                    ct1PPETensor = turbulencePPETensor1(NUmodes, NSUPmodes, NPmodes, nNutModes);
                }

                word ct2PPEStr = "ct2PPE_" + name(liftfield.size()) + "_" + name(
                                     NUmodes) + "_" + name(
                                     NSUPmodes) + "_" + name(NPmodes) + "_" + name(nNutModes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct2PPEStr))
                {
                    ITHACAstream::ReadDenseTensor(ct2PPETensor, "./ITHACAoutput/Matrices/",
                                                  ct2PPEStr);
                }
                else
                {
                    ct2PPETensor = turbulencePPETensor2(NUmodes, NSUPmodes, NPmodes, nNutModes);
                }

                word ct1PPEAveStr = "ct1PPEAve_" + name(liftfield.size()) + "_" + name(
                                        NUmodes) + "_" + name(
                                        NSUPmodes) + "_" + name(NPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct1PPEAveStr))
                { 
                    ITHACAstream::ReadDenseTensor(ct1PPEAveTensor, "./ITHACAoutput/Matrices/",
                                                  ct1PPEAveStr);
                }
                else
                {        
                    ct1PPEAveTensor = turbulencePPEAveTensor1(NUmodes, NSUPmodes, NPmodes);
                }

                word ct2PPEAveStr = "ct2PPEAve_" + name(liftfield.size()) + "_" + name(
                                        NUmodes) + "_" + name(
                                        NSUPmodes) + "_" + name(NPmodes) + "_t";

                if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + ct2PPEAveStr))
                {
                    ITHACAstream::ReadDenseTensor(ct2PPEAveTensor, "./ITHACAoutput/Matrices/",
                                                  ct2PPEAveStr);
                }
                else
                {
                    ct2PPEAveTensor = turbulencePPEAveTensor2(NUmodes, NSUPmodes, NPmodes);
                }
                if (bcMethod == "penalty")
                {
                    bcVelVec = bcVelocityVec(NUmodes, NSUPmodes);
                    bcVelMat = bcVelocityMat(NUmodes, NSUPmodes);
                }
            }
            else
            {
                B_matrix = diffusive_term(NUmodes, NPmodes, NSUPmodes);
                C_tensor = convective_term_tens(NUmodes, NPmodes, NSUPmodes);
                M_matrix = mass_term(NUmodes, NPmodes, NSUPmodes);
                K_matrix = pressure_gradient_term(NUmodes, NPmodes, NSUPmodes);
                D_matrix = laplacian_pressure(NPmodes);
                gTensor = divMomentum(NUmodes, NPmodes);
                BC1_matrix = pressure_BC1(NUmodes, NPmodes);
                bc2Tensor = pressureBC2(NUmodes, NPmodes);
                BC3_matrix = pressure_BC3(NUmodes, NPmodes);
                btMatrix = btTurbulence(NUmodes, NSUPmodes);
                ct1Tensor = turbulenceTensor1(NUmodes, NSUPmodes, nNutModes);
                ct2Tensor = turbulenceTensor2(NUmodes, NSUPmodes, nNutModes);
                ct1PPETensor = turbulencePPETensor1(NUmodes, NSUPmodes, NPmodes, nNutModes);
                ct2PPETensor = turbulencePPETensor2(NUmodes, NSUPmodes, NPmodes, nNutModes);
                ct1AveTensor = turbulenceAveTensor1(NUmodes, NSUPmodes);
                ct2AveTensor = turbulenceAveTensor2(NUmodes, NSUPmodes);
                ct1PPEAveTensor = turbulencePPEAveTensor1(NUmodes, NSUPmodes, NPmodes);
                ct2PPEAveTensor = turbulencePPEAveTensor2(NUmodes, NSUPmodes, NPmodes);
                
                if (bcMethod == "penalty")
                {
                    bcVelVec = bcVelocityVec(NUmodes, NSUPmodes);
                    bcVelMat = bcVelocityMat(NUmodes, NSUPmodes);
                }
            }

            // Export the matrices
            if (para->exportPython)
            {
                ITHACAstream::exportMatrix(B_matrix, "B", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(K_matrix, "K", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(D_matrix, "D", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(M_matrix, "M", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC1_matrix, "BC1", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC3_matrix, "BC3", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(C_tensor, "C", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(gTensor, "G", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(btMatrix, "bt", "python", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(bc2Tensor, "BC2", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1Tensor, "ct1", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2Tensor, "ct2", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1PPETensor, "ct1PPE", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2PPETensor, "ct2PPE", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1AveTensor, "ct1Ave", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2AveTensor, "ct2Ave", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1PPEAveTensor, "ct1PPEAve", "python",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2PPEAveTensor, "ct2PPEAve", "python",
                                           "./ITHACAoutput/Matrices/");
                // Matrices
                cnpy::save(B_matrix, "./ITHACAoutput/Matrices/B.npy");
                cnpy::save(K_matrix, "./ITHACAoutput/Matrices/K.npy");
                cnpy::save(D_matrix, "./ITHACAoutput/Matrices/D.npy");
                cnpy::save(M_matrix, "./ITHACAoutput/Matrices/M.npy");
                cnpy::save(BC1_matrix, "./ITHACAoutput/Matrices/BC1.npy");
                cnpy::save(BC3_matrix, "./ITHACAoutput/Matrices/BC3.npy");
                // Tensors
                cnpy::save(C_tensor, "./ITHACAoutput/Matrices/C.npy");
                cnpy::save(gTensor, "./ITHACAoutput/Matrices/G.npy");
                cnpy::save(bc2Tensor, "./ITHACAoutput/Matrices/BC2.npy");
                cnpy::save(ct1Tensor, "./ITHACAoutput/Matrices/ct1.npy");
                cnpy::save(ct2Tensor, "./ITHACAoutput/Matrices/ct2.npy");
                cnpy::save(ct1PPETensor, "./ITHACAoutput/Matrices/ct1PPE.npy");
                cnpy::save(ct2PPETensor, "./ITHACAoutput/Matrices/ct2PPE.npy");
                cnpy::save(ct1AveTensor, "./ITHACAoutput/Matrices/ct1Ave.npy");
                cnpy::save(ct2AveTensor, "./ITHACAoutput/Matrices/ct2Ave.npy"); 
                cnpy::save(ct1PPEAveTensor, "./ITHACAoutput/Matrices/ct1PPEAve.npy");
                cnpy::save(ct2PPEAveTensor, "./ITHACAoutput/Matrices/ct2PPEAve.npy");                               
            }
            

            if (para->exportMatlab)
            {
                ITHACAstream::exportMatrix(B_matrix, "B", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(K_matrix, "K", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(D_matrix, "D", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(M_matrix, "M", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC1_matrix, "BC1", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC3_matrix, "BC3", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(C_tensor, "C", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(gTensor, "G", "matlab", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(bc2Tensor, "BC2", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1Tensor, "ct1", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2Tensor, "ct2", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1PPETensor, "ct1PPE", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2PPETensor, "ct2PPE", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1AveTensor, "ct1Ave", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2AveTensor, "ct2Ave", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1PPEAveTensor, "ct1PPEAve", "matlab",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct2PPEAveTensor, "ct2PPEAve", "matlab",
                                           "./ITHACAoutput/Matrices/");
            }
            if (para->exportTxt)
            {
                ITHACAstream::exportMatrix(B_matrix, "B", "eigen", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(K_matrix, "K", "eigen", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(D_matrix, "D", "eigen", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(M_matrix, "M", "eigen", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC1_matrix, "BC1", "eigen",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportMatrix(BC3_matrix, "BC3", "eigen",
                                           "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(C_tensor, "C", "eigen",
                                           "./ITHACAoutput/Matrices/C");
                ITHACAstream::exportTensor(gTensor, "G", "eigen",
                                           "./ITHACAoutput/Matrices/G");
                ITHACAstream::exportTensor(bc2Tensor, "BC2_", "eigen",
                                           "./ITHACAoutput/Matrices/BC2");
                ITHACAstream::exportMatrix(btMatrix, "bt", "eigen", "./ITHACAoutput/Matrices/");
                ITHACAstream::exportTensor(ct1Tensor, "ct1_", "eigen",
                                           "./ITHACAoutput/Matrices/ct1");
                ITHACAstream::exportTensor(ct2Tensor, "ct2_", "eigen",
                                           "./ITHACAoutput/Matrices/ct2");
                ITHACAstream::exportTensor(ct1PPETensor, "ct1PPE_", "eigen",
                                           "./ITHACAoutput/Matrices/ct1PPE");
                ITHACAstream::exportTensor(ct2PPETensor, "ct2PPE_", "eigen",
                                           "./ITHACAoutput/Matrices/ct2PPE");
                ITHACAstream::exportTensor(ct1AveTensor, "ct1Ave_", "eigen",
                                           "./ITHACAoutput/Matrices/ct1Ave");
                ITHACAstream::exportTensor(ct2AveTensor, "ct2Ave_", "eigen",
                                           "./ITHACAoutput/Matrices/ct2Ave");
                ITHACAstream::exportTensor(ct1PPEAveTensor, "ct1PPEAve_", "eigen",
                                           "./ITHACAoutput/Matrices/ct1PPEAve");
                ITHACAstream::exportTensor(ct2PPEAveTensor, "ct2PPEAve_", "eigen",        
                                           "./ITHACAoutput/Matrices/ct2PPEAve");        
            }

            bTotalMatrix = B_matrix + btMatrix;
            label cSize = NUmodes + NSUPmodes + liftfield.size();
            cTotalTensor.resize(cSize, nNutModes, cSize);
            cTotalTensor = ct1Tensor + ct2Tensor;
            cTotalPPETensor.resize(NPmodes, nNutModes, cSize);
            cTotalPPETensor = ct1PPETensor + ct2PPETensor;
            cTotalAveTensor.resize(cSize, nutAve.size(), cSize);
            cTotalAveTensor = ct1AveTensor + ct2AveTensor;
            cTotalPPEAveTensor.resize(NPmodes, nutAve.size(), cSize);
            cTotalPPEAveTensor = ct1PPEAveTensor + ct2PPEAveTensor;
            
            
            // Export the matrix
            ITHACAstream::SaveDenseMatrix(coeffL2, "./ITHACAoutput/Matrices/",
                                          "coeffL2_nut_" + name(nNutModes));
            samples.resize(nNutModes);
            rbfSplines.resize(nNutModes);
            Eigen::MatrixXd weights;

            // for (label i = 0; i < nNutModes; i++)
            // {
            //     word weightName = "wRBF_N" + name(i + 1) + "_" + name(liftfield.size()) + "_"
            //                       + name(NUmodes) + "_" + name(NSUPmodes) ;

            //     if (ITHACAutilities::check_file("./ITHACAoutput/weightsSUP/" + weightName))
            //     {
            //         samples[i] = new SPLINTER::DataTable(1, 1);

            //         for (label j = 0; j < coeffL2.cols(); j++)
            //         {
            //             samples[i]->addSample(velRBF.row(j), coeffL2(i, j));
            //         }

            //         ITHACAstream::ReadDenseMatrix(weights, "./ITHACAoutput/weightsSUP/",
            //                                       weightName);
            //         rbfSplines[i] = new SPLINTER::RBFSpline(*samples[i],
            //                                                 SPLINTER::RadialBasisFunctionType::GAUSSIAN, weights, e);
            //         std::cout << "Constructing RadialBasisFunction for mode " << i + 1 << std::endl;
            //     }

            //     else
            //     {
            //         samples[i] = new SPLINTER::DataTable(1, 1);

            //         for (label j = 0; j < coeffL2.cols(); j++)
            //         {
            //             samples[i]->addSample(velRBF.row(j), coeffL2(i, j));
            //         }

            //         rbfSplines[i] = new SPLINTER::RBFSpline(*samples[i],
            //                                                 SPLINTER::RadialBasisFunctionType::GAUSSIAN, false, e);
            //         ITHACAstream::SaveDenseMatrix(rbfSplines[i]->weights,
            //                                       "./ITHACAoutput/weightsSUP/", weightName);
            //         std::cout << "Constructing RadialBasisFunction for mode " << i + 1 << std::endl;
            //     }
            // }
        }

};

int main(int argc, char *argv[])
{
    //return 0;
    tutorial22ppe example(argc, argv);
    word filename("./par");
    Eigen::VectorXd par;
    example.inletIndex.resize(1, 2);
    example.inletIndex << 0, 0;
    example.inletIndexT.resize(1, 1);
    example.inletIndexT << 1;
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
                             example._runTime());
    int NmodesU = para->ITHACAdict->lookupOrDefault<int>("NmodesU", 5);
    int NmodesP = para->ITHACAdict->lookupOrDefault<int>("NmodesP", 5);
    int NmodesSUP = para->ITHACAdict->lookupOrDefault<int>("NmodesSUP", 5);
    int NmodesNUT = para->ITHACAdict->lookupOrDefault<int>("NmodesNUT", 5);
    int NmodesProject = para->ITHACAdict->lookupOrDefault<int>("NmodesProject", 5);
    int NmodesMatrixRec = para->ITHACAdict->lookupOrDefault<int>("NmodesMatrixRec",
                          5);
    double penaltyFactor =
        para->ITHACAdict->lookupOrDefault<double>("penaltyFactor", 5);
    double U_BC = para->ITHACAdict->lookupOrDefault<double>("U_BC", 0.001);
    double romStartTime = para->ITHACAdict->lookupOrDefault<double>("romStartTime",
                          0);
    double romEndTime = para->ITHACAdict->lookupOrDefault<double>("romEndTime", 3);
    double romTimeStep = para->ITHACAdict->lookupOrDefault<double>("romTimeStep",
                         0.001);
    double e = para->ITHACAdict->lookupOrDefault<double>("RBFradius", 1);
    example.startTime = 79.992;
    example.finalTime = 99.996;
    example.timeStep = 0.0002;
    example.writeEvery = 0.004;
    example.offlineSolve();
    example.solvesupremizer();

    //modNut = ITHACAPOD::getModes(example.nutFields, example.nutModes, example._nut().name(),
    //                    example.podex, 0, 0, NmodesProject);
    ITHACAPOD::getModes(example.Ufield, example.Umodes, example._U().name(),
                        example.podex, 0, 0, NmodesProject);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(),
                        example.podex, 0, 0, NmodesProject);
    ITHACAPOD::getModes(example.supfield, example.supmodes, example._U().name(),
                        example.podex,
                        example.supex, 1, NmodesProject);
    
    example.projectPPE("./Matrices", NmodesU, NmodesP, NmodesSUP, NmodesNUT);

    Eigen::MatrixXd coeefs = ITHACAutilities::getCoeffs(example.Ufield, example.L_U_SUPmodes);
    Eigen::MatrixXd coeefsNut = ITHACAutilities::getCoeffs(example.nutFields, example.nutModes);
    Eigen::MatrixXd coeefsP = ITHACAutilities::getCoeffs(example.Pfield, example.Pmodes);

    cnpy::save(coeefs, "./ITHACAoutput/Matrices/coeefs.npy");
    cnpy::save(coeefsNut, "./ITHACAoutput/Matrices/coeefsNut.npy");
    cnpy::save(coeefsP, "./ITHACAoutput/Matrices/coeefsP.npy");

}
