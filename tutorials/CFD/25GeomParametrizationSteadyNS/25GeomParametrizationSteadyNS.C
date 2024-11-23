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
Description
    Example of turbulent steady NS Reduction Problem solved by the use of the SIMPLE algorithm
SourceFiles
    25GeomParametrizationSteadyNS.C
\*---------------------------------------------------------------------------*/

#include "SteadyNSSimple.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "ReducedSimpleSteadyNS.H"
#include "forces.H"
#include "IOmanip.H"
#include "RBFMotionSolver.H"
#include "SteadyNSTurb.H"
#include "SteadyNSTurbIntrusive.H"
#include <Eigen/Dense>


class SteadyNSSimpleNN : public SteadyNSSimple
{
    public:
        /// Constructor
        SteadyNSSimpleNN(int argc, char* argv[])
            :
            SteadyNSSimple(argc, argv)
        {
        };

        // This function computes the coefficients which are later used for training
        void getTurbNN()
        {
            if (!ITHACAutilities::check_folder("ITHACAoutput/NN/coeffs"))
            {
                mkDir("ITHACAoutput/NN/coeffs");
                // Read Fields for Train
                PtrList<volVectorField> UfieldTrain;
                PtrList<volScalarField> PfieldTrain;
                PtrList<volScalarField> nutFieldsTrain;
                ITHACAstream::readMiddleFields(UfieldTrain, _U(),
                                               "./ITHACAoutput/Offline/");
                ITHACAstream::readMiddleFields(PfieldTrain, _p(),
                                               "./ITHACAoutput/Offline/");
                auto nut_train = _mesh().lookupObject<volScalarField>("nut");
                ITHACAstream::readMiddleFields(nutFieldsTrain, nut_train,
                                               "./ITHACAoutput/Offline/");
                /// Compute the coefficients for train
                std::cout << "Computing the coefficients for U train" << std::endl;
                Eigen::MatrixXd coeffL2U_train = ITHACAutilities::getCoeffs(UfieldTrain,
                                                 Umodes,
                                                 0, true);
                std::cout << "Computing the coefficients for p train" << std::endl;
                Eigen::MatrixXd coeffL2P_train = ITHACAutilities::getCoeffs(PfieldTrain,
                                                 Pmodes,
                                                 0, true);
                std::cout << "Computing the coefficients for nuT train" << std::endl;
                Eigen::MatrixXd coeffL2Nut_train = ITHACAutilities::getCoeffs(nutFieldsTrain,
                                                   nutModes,
                                                   0, true);
                coeffL2U_train.transposeInPlace();
                coeffL2P_train.transposeInPlace();
                coeffL2Nut_train.transposeInPlace();
                cnpy::save(coeffL2U_train, "ITHACAoutput/NN/coeffs/coeffL2UTrain.npy");
                cnpy::save(coeffL2P_train, "ITHACAoutput/NN/coeffs/coeffL2PTrain.npy");
                cnpy::save(coeffL2Nut_train, "ITHACAoutput/NN/coeffs/coeffL2NutTrain.npy");
                // Read Fields for Test
                PtrList<volVectorField> UfieldTest;
                PtrList<volScalarField> PfieldTest;
                PtrList<volScalarField> nutFieldsTest;
                /// Compute the coefficients for test
                ITHACAstream::readMiddleFields(UfieldTest, _U(),
                                               "./ITHACAoutput/checkOff/");
                ITHACAstream::readMiddleFields(PfieldTest, _p(),
                                               "./ITHACAoutput/checkOff/");
                auto nut_test = _mesh().lookupObject<volScalarField>("nut");
                ITHACAstream::readMiddleFields(nutFieldsTest, nut_test,
                                               "./ITHACAoutput/checkOff/");
                // Compute the coefficients for test
                Eigen::MatrixXd coeffL2U_test = ITHACAutilities::getCoeffs(UfieldTest,
                                                Umodes,
                                                0, true);
                Eigen::MatrixXd coeffL2P_test = ITHACAutilities::getCoeffs(PfieldTest,
                                                Pmodes,
                                                0, true);
                Eigen::MatrixXd coeffL2Nut_test = ITHACAutilities::getCoeffs(nutFieldsTest,
                                                  nutModes,
                                                  0, true);
                coeffL2U_test.transposeInPlace();
                coeffL2P_test.transposeInPlace();
                coeffL2Nut_test.transposeInPlace();
                cnpy::save(coeffL2U_test, "ITHACAoutput/NN/coeffs/coeffL2UTest.npy");
                cnpy::save(coeffL2P_test, "ITHACAoutput/NN/coeffs/coeffL2PTest.npy");
                cnpy::save(coeffL2Nut_test, "ITHACAoutput/NN/coeffs/coeffL2NutTest.npy");
            }
        }

};

class tutorial01cl : public SteadyNSSimpleNN
{
    public:
        /// Constructor
        explicit tutorial01cl(int argc, char* argv[])
            :
            SteadyNSSimpleNN(argc, argv)
        {
            curX = _mesh().points();
            point0 = _mesh().points();
        }

        /// Initial coordinates of the grid points
        vectorField point0;
        /// List to store the moved coordinates
        List<vector> curX;

        /// Perform an Offline solve
        void offlineSolve(Eigen::MatrixXd Box, List<label> patches,
                          word folder = "./ITHACAoutput/Offline/")
        {
            //Vector<double> inl(0, 0, 0);
            List<scalar> mu_now(3);
            volVectorField& U = _U();
            volScalarField& p = _p();

            // if the offline solution is already performed read the fields
            if (offline && ITHACAutilities::isTurbulent() &&
                    !ITHACAutilities::check_folder("./ITHACAoutput/POD/1"))
            {
                ITHACAstream::readMiddleFields(Ufield, U, folder);
                ITHACAstream::readMiddleFields(Pfield, p, folder);
                auto nut = _mesh().lookupObject<volScalarField>("nut");
                ITHACAstream::readMiddleFields(nutFields, nut, folder);
            }
            else if (offline && !ITHACAutilities::check_folder("./ITHACAoutput/POD/1"))
            {
                ITHACAstream::readMiddleFields(Ufield, U, folder);
                ITHACAstream::readMiddleFields(Pfield, p, folder);
            }
            else if (!offline)
            {
                //Vector<double> Uinl(1, 0, 0);
                for (label i = 0; i < mu.rows(); i++)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        mu_now[k] = mu(i, k);
                    }
                    _mesh().movePoints(point0);
                    List<vector> points2Move;
                    labelList boxIndices = ITHACAutilities::getIndicesFromBox(_mesh(), patches, Box,
                                           points2Move);
                    linearMovePts(mu_now[0], mu_now[1], mu_now[2], points2Move);

                    for (int j = 0; j < boxIndices.size(); j++)
                    {
                        curX[boxIndices[j]] = points2Move[j];
                    }
                    Field<vector> curXV(curX);
                    _mesh().movePoints(curXV);
                    ITHACAstream::writePoints(_mesh().points(), folder,
                                              name(i + 1) + "/polyMesh/");
                    truthSolve2(mu_now, folder);
                    int middleF = 1;

                    while (ITHACAutilities::check_folder(folder + name(
                            i + 1) + "/" + name(middleF)))
                    {
                        word command("ln -s  $(readlink -f " + folder + name(
                                         i + 1) + "/polyMesh/) " + folder + name(i + 1) + "/" + name(
                                         middleF) + "/" + " >/dev/null 2>&1");
                        std::cout.setstate(std::ios_base::failbit);
                        system(command);
                        std::cout.clear();
                        middleF++;
                    }
                    restart();
                }
            }
        }

        void linearMovePts(double angle,
                double h1_new,
                double h2_new,
                List<vector>& points2Move)
        {
            double sMax;
            scalarList x;
            scalarList y;

            for (label i = 0; i < points2Move.size(); i++)
            {
                x.append(points2Move[i].component(0));
                y.append(points2Move[i].component(1));
            }

            double xMin = min(x);
            double xMax = max(x);
            double yMin = min(y);
            double yMax = max(y);
            // initialize h1 and h2 and quotients of new h1/h2 w.r.t. old ones
            double h2 = yMax - yMin;
            double h1 = abs(yMax);
            double r_h2 = h2_new / h2;
            double r_h1 = h1_new / h1;
            double r_h2_h1 = (h2_new-h1_new) / (h2-h1);

            double yMax_down = 0.0;
            double yMin_down = yMin;
            double xMin_down = 2.0;
            double xMax_down = 7.0;

            sMax = (yMax_down - yMin_down) * std::tan(M_PI * angle / 180);

            double diff_step = 0.3;
            double shift = h1 - diff_step;
            double step = h2 - 2 * diff_step;
            double zero_distance_right = abs(yMin_down) - diff_step;


            scalarList ytmp;
            scalarList xtmp;

            for (label i = 0; i < points2Move.size(); i++)
            {
                //deformation by h1 in top region
                if (points2Move[i].component(0) >= xMin &&
                    points2Move[i].component(1) >= yMax_down &&
                    points2Move[i].component(1) <= yMax)
                {
                    points2Move[i].component(1) = points2Move[i].component(1) * r_h1;

                }
                // deformation by (h2-h1) and mu (angle) in bottom region
                if (points2Move[i].component(0) >= xMin_down &&
                    points2Move[i].component(1) <= yMax_down)
                {
                    points2Move[i].component(1) = points2Move[i].component(1) * r_h2_h1;
                    // linear deformation by the angle only in the interested left region
                    if (points2Move[i].component(0) <= xMax_down)
                    {
                        points2Move[i].component(0) = points2Move[i].component(0) +
                                                      (yMax_down - points2Move[i].component(1)) / (yMax_down - yMin_down) * (xMax_down -
                                                              points2Move[i].component(0)) / (xMax_down - xMin_down) * sMax;
                    }
                }
                // deformation by h2 in right region
                if (points2Move[i].component(0) > xMax_down)
                    {
                        points2Move[i].component(1) = points2Move[i].component(1) * ((r_h2 * step/(r_h2_h1 * zero_distance_right + shift * r_h1) -1)*(points2Move[i].component(0) - xMax_down)/(xMax - xMax_down) + 1);
                    if (points2Move[i].component(0) == 9)
                        {
                            ytmp.append(points2Move[i].component(1));
                        }
                    }
            }
            // extract ymax and ymin at outlet (xtmp=9) and center the deformation in the right region
            double ymax_out = max(ytmp);
            double ymin_out = min(ytmp);
            double translation1 = (ymin_out + ymax_out)/2;
            double translation2 = (2*h1_new - h2_new)/2;
            double translation = - translation1 + translation2;
            for (label i = 0; i < points2Move.size(); i++)
            {
                // deformation by h2 in right region
                if (points2Move[i].component(0) > xMax_down)
                    {
                        points2Move[i].component(1) = points2Move[i].component(1) + translation*(points2Move[i].component(0) - xMax_down)/(xMax - xMax_down);
                    }
            }
        }
};

int main(int argc, char* argv[])
{
    // Construct the tutorial object
    tutorial01cl example(argc, argv);
    // Read some parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
                             example._runTime());
    // Read the files where the parameters are stored
    std::ifstream exFileOff("./paramOff_mat.txt");

    double angMax = 75;
    double angMin = 0;
    double h1Min = 1.;
    double h1Max = 1.5;
    double h2Min = 1.8;
    double h2Max = 2.3;

    if (exFileOff)
    {
        example.mu  = ITHACAstream::readMatrix("./paramOff_mat.txt");
    }
    else
    {
        example.mu = Eigen::MatrixXd(50, 3);
        example.mu.col(0) = ITHACAutilities::rand(50, 1, angMin, angMax);
        example.mu.col(1) = ITHACAutilities::rand(50, 1, h1Min, h1Max);
        example.mu.col(2) = ITHACAutilities::rand(50, 1, h2Min, h2Max);
        ITHACAstream::exportMatrix(example.mu, "paramOff", "eigen", "./");
    }

    Eigen::MatrixXd paramOn;
    std::ifstream exFileOn("./paramOn_mat.txt");

    if (exFileOn)
    {
        paramOn = ITHACAstream::readMatrix("./paramOn_mat.txt");
    }
    else
    {
        paramOn = Eigen::MatrixXd(10, 3);
        paramOn.col(0) = ITHACAutilities::rand(10, 1, angMin, angMax);
        paramOn.col(1) = ITHACAutilities::rand(10, 1, h1Min, h1Max);
        paramOn.col(2) = ITHACAutilities::rand(10, 1, h2Min, h2Max);
        ITHACAstream::exportMatrix(paramOn, "paramOn", "eigen", "./");
    }

    //Set the box including the step
    Eigen::MatrixXd Box(2, 3);
    Box << -0.01, 1.01, 0.11,
        9.02, -0.666669, -0.01;
    //Select the patches to be moved
    List<label> movPat;
    movPat.append(0);
    movPat.append(1);
    movPat.append(2);
    movPat.append(3);
    movPat.append(4);
    movPat.append(5);
    // Set the maximum iterations number for the offline phase
    example.maxIter = para->ITHACAdict->lookupOrDefault<int>("maxIter", 2000);
    // Perform the offline solve
    example.offlineSolve(Box, movPat);
    List<vector> points2Move;
    labelList boxIndices = ITHACAutilities::getIndicesFromBox(example._mesh(),
                           movPat, Box,
                           points2Move);
    example.linearMovePts(37.5, 1.25, 2.05, points2Move);
    for (int j = 0; j < boxIndices.size(); j++)
    {
        example.curX[boxIndices[j]] = points2Move[j];
    }

    Field<vector> curXV(example.curX);
    example._mesh().movePoints(curXV);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Ufield, example.Umodes, example._U().name(),
                        example.podex, 0, 0,
                        example.NUmodesOut);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(),
                        example.podex, 0, 0,
                        example.NPmodesOut);
    // Error analysis
    tutorial01cl checkOff(argc, argv);
    std::clock_t startOff;
    double durationOff;
    startOff = std::clock();

    if (!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    {
        checkOff.restart();
        ITHACAparameters* para = ITHACAparameters::getInstance(checkOff._mesh(),
                                 checkOff._runTime());
        checkOff.offline = false;
        checkOff.mu = paramOn;
        checkOff.offlineSolve(Box, movPat, "./ITHACAoutput/checkOff/");
        checkOff.offline = true;
    }

    durationOff = (std::clock() - startOff);

    if (ITHACAutilities::isTurbulent())
    {
        ITHACAPOD::getModes(example.nutFields, example.nutModes, "nut",
                            example.podex, 0, 0, example.NNutModesOut);
        // Create the RBF for turbulence
        example.getTurbNN();
    }

//    std::cout << "The offline phase duration is equal to " << durationOff <<
//              std::endl;
    exit(0);
}
