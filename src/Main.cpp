////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Main.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <iostream>

#include "BlasHelpers.h"
#include "ForLoopConvolution.h"
#include "PaddedConvolution.h"
#include "UnrolledConvolution.h"
#include "Tensor.h"

int main(int argc, char** argv)
{
    // define parameters
    int wRows = 3;
    int wCols = 3;
    int wChls = 2;
    int wCount = 3;

    int yRows = 4;
    int yCols = 4;
    int yChls = wCount;

    int vStride = 1;
    int hStride = 1;

    int xVPad = (wRows - 1) / 2; // on each side (top and bottom)
    int xHPad = (wCols - 1) / 2; // on each side (left and right)

    int xRows = (yRows - 1) * vStride + wRows; // includes any input padding
    int xCols = (yCols - 1) * hStride + wCols; // includes any input padding
    int xChls = wChls;

    int xIntRows = xRows - 2 * xVPad; // excludes any input padding
    int xIntCols = xCols - 2 * xHPad; // excludes any input padding
    int xIntChls = xChls;

    // random seeds and engine
    std::seed_seq seed1 = {103, 311, 1283};
    std::seed_seq seed2 = {3929, 437, 859};
    std::default_random_engine engine;

    // generate random filters
    engine.seed(seed1);
    auto WRowMaj = GetRandomTensor<float, 4>(engine, { wCount, wRows, wCols, wChls }, RowMaj4Order);
 
    engine.seed(seed1);
    auto WChlMaj = GetRandomTensor<float, 4>(engine, { wCount, wRows, wCols, wChls }, ChlMaj4Order);
 
    // generate the same input in both row and channel major orders, and with both exp and imp padding
    engine.seed(seed2);
    auto XRowMajExp = GetRandomTensor<float, 3>(engine, { xRows, xCols, xChls }, RowMaj3Order, {xVPad, xHPad, 0});

    engine.seed(seed2);
    auto XChlMajExp = GetRandomTensor<float, 3>(engine, { xRows, xCols, xChls }, ChlMaj3Order, {xVPad, xHPad, 0});

    engine.seed(seed2);
    auto XRowMajImp = GetRandomTensor<float, 3>(engine, { xIntRows, xIntCols, xIntChls }, RowMaj3Order);

    engine.seed(seed2);
    auto XChlMajImp = GetRandomTensor<float, 3>(engine, { xIntRows, xIntCols, xIntChls }, ChlMaj3Order);

    // for loop convolution
    auto Y0 = Tensor<float,3> ({ yRows, yCols, yChls }, RowMaj3Order);
    ForLoopConvolution(WRowMaj.Data(), XRowMajExp.Data(), Y0.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    std::cout << Y0 << std::endl << std::endl;

    // unrolled convolutions
    auto Y1 = Tensor<float,3> ({ yRows, yCols, yChls }, RowMaj3Order);
    UnrolledConvolutionRowMaj(WRowMaj.Data(), XRowMajExp.Data(), Y1.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    std::cout << Y1 << std::endl << std::endl;

    auto Y2 = Tensor<float,3> ({ yRows, yCols, yChls }, RowMaj3Order);
    UnrolledConvolutionChlMaj(WRowMaj.Data(), XChlMajExp.Data(), Y2.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    std::cout << Y2 << std::endl << std::endl;

    // padded convolutions
    auto Y3 = Tensor<float,3> ({ yRows, yCols, yChls }, RowMaj3Order);
    UnrolledConvolutionChlMaj(WRowMaj.Data(), XChlMajImp.Data(), Y3.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    std::cout << Y3 << std::endl << std::endl;


    return 0;
}