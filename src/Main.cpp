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
#include "ConvolutionProperties.h"
#include "ForLoopConvolution.h"
#include "PaddedConvolution.h"
#include "UnrolledConvolution.h"
#include "Tensor.h"
#include "TestHelpers.h"

int main(int argc, char** argv)
{
    // filter shape
    int wRows = 3;
    int wCols = 3;
    int wChls = 4;
    int wCount = 3;

    // output shape
    int yRows = 6;
    int yCols = 5;
    int yChls = wCount;

    // convolution strides
    int vStride = 1;
    int hStride = 1;

    // input shape
    int xRows = (yRows - 1) * vStride + wRows; // includes any input padding
    int xCols = (yCols - 1) * hStride + wCols; // includes any input padding
    int xChls = wChls;

    // input padding 
    int xPadTop = (wRows - 1) / 2;
    int xPadBottom = wRows - 1 - xPadTop;
    int xPadLeft = (wCols - 1) / 2;
    int xPadRight = wCols - 1 - xPadLeft; 

    // random seeds and engine
    std::seed_seq seed1 = {103, 311, 1283};
    std::seed_seq seed2 = {3929, 437, 859};
    std::default_random_engine engine;

    // generate random filters in two memory orders
    engine.seed(seed1);
    auto WFilMaj = GetRandomTensor<float, 4>(engine, { wCount, wRows, wCols, wChls }, {3, 2, 1, 0});
    engine.seed(seed1);
    auto WRowMaj = GetRandomTensor<float, 4>(engine, { wCount, wRows, wCols, wChls }, {0, 3, 2, 1});

    // generate random input in both row-major and channel-major orders, and with both explicit and implicit zero-padding
    engine.seed(seed2);
    auto XRowMajExp = GetRandomTensor<float, 3>(engine, { xRows, xCols, xChls }, RowMaj3Order, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XChlMajExp = GetRandomTensor<float, 3>(engine, { xRows, xCols, xChls }, ChlMaj3Order, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XRowMajImp = GetRandomTensor<float, 3>(engine, { yRows, yCols, xChls }, RowMaj3Order);
    engine.seed(seed2);
    auto XChlMajImp = GetRandomTensor<float, 3>(engine, { yRows, yCols, xChls }, ChlMaj3Order);

    // debug

    // auto WFilMaj = GetTensor4<float>(
    //   { { { {1, 1}, {-1, 1}, {3, 1} },
    //       { {2, 2}, {2, 3}, {2, 2} },
    //       { {3, 1}, {3, -4}, {-4, -3} } },
    //     { { {-1, 1}, {1, -1}, {-1, 2} },
    //       { {1, 2}, {3, 2}, {2, 2} },
    //       { {3, 3}, {-3, 3}, {-3, 3} } },
    //     { { {1, 1}, {2, 1}, {4, -2} },
    //       { {2, -2}, {4, 2}, {1, 2} },
    //       { {3, 1}, {-3, 1}, {4, 3} } } },
    //       {3, 2, 1, 0});
 
    // auto WRowMaj = GetTensor4<float>(
    //   { { { {1, 1}, {-1, 1}, {3, 1} },
    //       { {2, 2}, {2, 3}, {2, 2} },
    //       { {3, 1}, {3, -4}, {-4, -3} } },
    //     { { {-1, 1}, {1, -1}, {-1, 2} },
    //       { {1, 2}, {3, 2}, {2, 2} },
    //       { {3, 3}, {-3, 3}, {-3, 3} } },
    //     { { {1, 1}, {2, 1}, {4, -2} },
    //       { {2, -2}, {4, 2}, {1, 2} },
    //       { {3, 1}, {-3, 1}, {4, 3} } } },
    //       {0, 3, 2, 1});
 

    // auto XRowMajExp = GetTensor3<float>(
    //     { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    //       { {0, 0}, {11, 12}, {13, 14}, {1, 2}, {15, 16}, {0, 0} },
    //       { {0, 0}, {2, 2}, {2, 2}, {1, 2}, {2, 2}, {0, 0} },
    //       { {0, 0}, {3, 2}, {3, 4}, {1, 2}, {5, 6}, {0, 0} },
    //       { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } },
    //       RowMaj3Order);

    // auto XChlMajExp = GetTensor3<float>(
    //     { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    //       { {0, 0}, {11, 12}, {13, 14}, {1, 2}, {15, 16}, {0, 0} },
    //       { {0, 0}, {2, 2}, {2, 2}, {1, 2}, {2, 2}, {0, 0} },
    //       { {0, 0}, {3, 2}, {3, 4}, {1, 2}, {5, 6}, {0, 0} },
    //       { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } },
    //       ChlMaj3Order);

    // auto XRowMajImp = GetTensor3<float>(
    //     { { {11, 12}, {13, 14}, {1, 2}, {15, 16} },
    //       { {2, 2}, {2, 2}, {1, 2}, {2, 2} },
    //       { {3, 2}, {3, 4}, {1, 2}, {5, 6} } },
    //       RowMaj3Order);

    // auto XChlMajImp = GetTensor3<float>(
    //     { { {11, 12}, {13, 14}, {1, 2}, {15, 16} },
    //       { {2, 2}, {2, 2}, {1, 2}, {2, 2} },
    //       { {3, 2}, {3, 4}, {1, 2}, {5, 6} } },
    //       ChlMaj3Order);

    // for loop convolution
    std::cout << "for loop convolution" << std::endl;
    RunTest([&]()
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        return YRowMaj;
    });

    // unrolled convolutions
    std::cout << "unrolled convolutions" << std::endl;
    RunTest([&]()
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledOutput>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        return YRowMaj;
    });

    // padded convolutions
    std::cout << "padded convolutions" << std::endl;
    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitStride, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        std::vector<float> space(9 * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitStride, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitStride, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        return YRowMaj;
    });

    // partially unrolled convolutions
    std::cout << "partially unrolled convolutions" << std::endl;
    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitStride>{};
        auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
        std::vector<float> space(yRows * yCols * wChls);
        Convolution(properties, WRowMaj.Data(), XRowMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitStride>{};
        auto YRowMaj = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        return YRowMaj;
    });

    RunTest([&]()
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, RowMajorFilters, RowMajorOutput, UnitStride, UnrolledInput>{};
        auto YRowMaj = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        return YRowMaj;
    });



    return 0;
}