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

void RunAllTests(int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int xCount, int vStride, int hStride)
{
    // output shape
    int yChls = wCount;

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
    
    // allocate output tensors
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3Order);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3Order);

    // for loop convolution
    std::cout << "for loop convolution" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        std::cout << YRowMaj << std::endl << std::endl;
    }

    // unrolled convolutions
    std::cout << "unrolled convolutions" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        std::vector<float> space(xRows * xCols * wCount * wRows * wCols);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YChlMaj << std::endl << std::endl;
    }

    // padded convolutions
    std::cout << "padded convolutions" << std::endl;
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space(9 * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    // partially unrolled convolutions
    std::cout << "partially unrolled convolutions" << std::endl;
    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        std::vector<float> space(yRows * yCols * wChls);
        Convolution(properties, WRowMaj.Data(), XRowMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        std::cout << YRowMajExp << std::endl << std::endl;
    }
}

void RunDebugTests()
{
    // filter shape
    int wRows = 3;
    int wCols = 3;
    int wChls = 2;
    int wCount = 3;

    // output shape
    int yRows = 3;
    int yCols = 4;
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
    
    // allocate output tensors
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3Order);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3Order);

    auto WFilMaj = GetTensor4<float>(
      { { { {1, 1}, {-1, 1}, {3, 1} },
          { {2, 2}, {2, 3}, {2, 2} },
          { {3, 1}, {3, -4}, {-4, -3} } },
        { { {-1, 1}, {1, -1}, {-1, 2} },
          { {1, 2}, {3, 2}, {2, 2} },
          { {3, 3}, {-3, 3}, {-3, 3} } },
        { { {1, 1}, {2, 1}, {4, -2} },
          { {2, -2}, {4, 2}, {1, 2} },
          { {3, 1}, {-3, 1}, {4, 3} } } },
          {3, 2, 1, 0});
 
    std::cout << "Filter-Major W" << std::endl;
    std::cout << WFilMaj << std::endl << std::endl;

    auto WRowMaj = GetTensor4<float>(
      { { { {1, 1}, {-1, 1}, {3, 1} },
          { {2, 2}, {2, 3}, {2, 2} },
          { {3, 1}, {3, -4}, {-4, -3} } },
        { { {-1, 1}, {1, -1}, {-1, 2} },
          { {1, 2}, {3, 2}, {2, 2} },
          { {3, 3}, {-3, 3}, {-3, 3} } },
        { { {1, 1}, {2, 1}, {4, -2} },
          { {2, -2}, {4, 2}, {1, 2} },
          { {3, 1}, {-3, 1}, {4, 3} } } },
          {0, 3, 2, 1});

    std::cout << "Row-Major W" << std::endl;
    std::cout << WRowMaj << std::endl << std::endl;

    auto XRowMajExp = GetTensor3<float>(
        { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
          { {0, 0}, {11, 12}, {13, 14}, {1, 2}, {15, 16}, {0, 0} },
          { {0, 0}, {2, 2}, {2, 2}, {1, 2}, {2, 2}, {0, 0} },
          { {0, 0}, {3, 2}, {3, 4}, {1, 2}, {5, 6}, {0, 0} },
          { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } },
          RowMaj3Order);

    std::cout << "Row-Major Explicitly-Padded X" << std::endl;
    std::cout << XRowMajExp << std::endl << std::endl;

    auto XChlMajExp = GetTensor3<float>(
        { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
          { {0, 0}, {11, 12}, {13, 14}, {1, 2}, {15, 16}, {0, 0} },
          { {0, 0}, {2, 2}, {2, 2}, {1, 2}, {2, 2}, {0, 0} },
          { {0, 0}, {3, 2}, {3, 4}, {1, 2}, {5, 6}, {0, 0} },
          { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } },
          ChlMaj3Order);

    std::cout << "Channel-Major Explicitly-Padded X" << std::endl;
    std::cout << XChlMajExp << std::endl << std::endl;

    auto XRowMajImp = GetTensor3<float>(
        { { {11, 12}, {13, 14}, {1, 2}, {15, 16} },
          { {2, 2}, {2, 2}, {1, 2}, {2, 2} },
          { {3, 2}, {3, 4}, {1, 2}, {5, 6} } },
          RowMaj3Order);

    std::cout << "Row-Major Implicitly-Padded X" << std::endl;
    std::cout << XRowMajImp << std::endl << std::endl;

    auto XChlMajImp = GetTensor3<float>(
        { { {11, 12}, {13, 14}, {1, 2}, {15, 16} },
          { {2, 2}, {2, 2}, {1, 2}, {2, 2} },
          { {3, 2}, {3, 4}, {1, 2}, {5, 6} } },
          ChlMaj3Order);

    std::cout << "Channel-Major Implicitly-Padded X" << std::endl;
    std::cout << XChlMajImp << std::endl << std::endl;

    // for loop convolution
    std::cout << "forLoop convolution" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        std::cout << YRowMaj << std::endl << std::endl;
    }

    // unrolled convolutions
    std::cout << "rowMajUnrolledInput convolution" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "chlMajUnrolledInput convolution" << std::endl;
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        std::vector<float> space(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "unrolledOutput convolution" << std::endl;
    {
        auto properties = ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        std::vector<float> space(xRows * xCols * wCount * wRows * wCols);
        Convolution(properties, WFilMaj.Data(), XRowMajExp.Data(), YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YChlMaj << std::endl << std::endl;
    }

    // padded convolutions
    std::cout << "padded convolutions" << std::endl;
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space(9 * wChls * yRows * yCols);
        Convolution(properties, WFilMaj.Data(), XChlMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        std::vector<float> space((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj.Data(), XChlMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    // partially unrolled convolutions
    std::cout << "partially unrolled convolutions" << std::endl;
    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        std::vector<float> space(yRows * yCols * wChls);
        Convolution(properties, WRowMaj.Data(), XRowMajImp.Data(), YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj.Data(), XRowMajExp.Data(), YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        std::cout << YRowMajExp << std::endl << std::endl;
    }
}

void PrintBenchmarkNames()
{
    std::cout << "forLoop, ";
    std::cout << "rowMajUnrolledInput, ";
    std::cout << "chlMajUnrolledInput, ";
    std::cout << "UnrolledOutput";
}

void RunAllBenchmarks(double testDuration, int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int xCount, int vStride, int hStride)
{
    // output shape
    int yChls = wCount;

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
    auto XRowMajExp = GetRandomTensors<float, 3>(xCount, engine, { xRows, xCols, xChls }, RowMaj3Order, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XChlMajExp = GetRandomTensors<float, 3>(xCount, engine, { xRows, xCols, xChls }, ChlMaj3Order, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XRowMajImp = GetRandomTensors<float, 3>(xCount, engine, { yRows, yCols, xChls }, RowMaj3Order);
    engine.seed(seed2);
    auto XChlMajImp = GetRandomTensors<float, 3>(xCount, engine, { yRows, yCols, xChls }, ChlMaj3Order);

    // allocate output tensors
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3Order);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3Order);

    // for loop
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        });
        std::cout << time << ", ";
    }

    // row major unrolled input
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }

    // channel major unrolled input
    if(hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
        auto time = RunBenchmark<float>(testDuration, XChlMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // unrolled output
    {
        auto properties = ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        auto space = std::vector<float>(xRows * xCols * wCount * wRows * wCols);
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }

    // padded
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>(9 * wChls * yRows * yCols);
        auto time = RunBenchmark<float>(testDuration, XChlMajImp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        });
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XChlMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        });
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XChlMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        });
    }

    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        auto space = std::vector<float>(yRows * yCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XRowMajImp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        });
    }

    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        });
    }

    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        });
    }
}

int main(int argc, char** argv)
{
    // filter shape
    int wRows = 3;
    int wCols = 3;
    int wChls = 2;
    int wCount = 3;

    // output shape
    int yRows = 3;
    int yCols = 3;

    // convolution strides
    int vStride = 1;
    int hStride = 1;

    // input shape
    int xCount = 10;

    double testDuration = 1000;

    RunDebugTests();

    return 0;
}

