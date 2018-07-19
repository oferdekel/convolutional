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
#include "Tensor.h"

int main(int argc, char** argv)
{
    // define parameters
    size_t wRows = 3;
    size_t wCols = 3;
    size_t wChls = 2;
    size_t wCount = 3;

    size_t yRows = 4;
    size_t yCols = 4;
    size_t yChls = wCount;

    size_t vStride = 1;
    size_t hStride = 1;

    size_t xVPad = (wRows - 1) / 2; // on each side (top and bottom)
    size_t xHPad = (wCols - 1) / 2; // on each side (left and right)

    size_t xRows = (yRows - 1) * vStride + wRows; // includes any input padding
    size_t xCols = (yCols - 1) * hStride + wCols; // includes any input padding
    size_t xChls = wChls;

    size_t xIntRows = xRows - 2 * xVPad; // excludes any input padding
    size_t xIntCols = xCols - 2 * xHPad; // excludes any input padding
    size_t xIntChls = xChls;

    // random seeds and engine
    std::seed_seq seed1 = {103, 311, 1283};
    std::seed_seq seed2 = {3929, 437, 859};
    std::default_random_engine engine;

    // generate random filters
    engine.seed(seed1);
    auto W = GetRandomTensor<float>(engine, { wCount, wRows, wCols, wChls }, RowMaj4Order);
 
    // generate the same input in both row and channel major orders, and with both exp and imp padding
    engine.seed(seed2);
    auto XRowMajExp = GetRandomTensor<float>(engine, { xRows, xCols, xChls }, RowMaj3Order, {xVPad, xHPad, 0});

    engine.seed(seed2);
    auto XChlMajExp = GetRandomTensor<float>(engine, { xRows, xCols, xChls }, ChlMaj3Order, {xVPad, xHPad, 0});

    engine.seed(seed2);
    auto XRowMajImp = GetRandomTensor<float>(engine, { xIntRows, xIntCols, xIntChls }, RowMaj3Order);

    engine.seed(seed2);
    auto XChlMajImp = GetRandomTensor<float>(engine, { xIntRows, xIntCols, xIntChls }, ChlMaj3Order);

    // for loop convolution
    auto Y0 = Tensor<float,3> ({ yRows, yCols, yChls }, RowMaj3Order);
    ForLoopConvolution(W.Data(), XRowMajExp.Data(), Y0.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);

    std::cout << Y0 << std::endl;


    return 0;
}