////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Convolutions
//  File:     Main.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <iostream>

#include "BlasHelpers.h"
#include "Matrix.h"
#include "Tensor.h"

int main(int argc, char** argv)
{
    std::default_random_engine engine;
    std::normal_distribution<float> normal(0, 1);
    auto rng = [&](){ return normal(engine);};

    Matrix<float> A({{1,2,3},{3,2,1}}, MatrixOrder::RowMajor);
    Matrix<float> B({{1, 1},{2, 2},{1, 0}}, MatrixOrder::RowMajor);
    Matrix<float> C(2, 2, MatrixOrder::RowMajor);
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;

    Gemm(A, B, C);

    std::cout << C << std::endl;

    return 0;

}