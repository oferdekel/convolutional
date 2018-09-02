////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Main.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "CSVParser.h"
#include "ForLoopConvolution.h"
#include "PartiallyUnrolledInputExplicitOutPaddingConvolution.h"
#include "PartiallyUnrolledInputExplicitPaddingConvolution.h"
#include "PartiallyUnrolledInputImplicitInPaddingConvolution.h"
#include "Tensor.h"
#include "TestHelpers.h"
#include "UnrolledInputChlMajInputConvolution.h"
#include "UnrolledInputConvolution.h"
#include "UnrolledInputExplicitOutPaddingConvolution.h"
#include "UnrolledInputExplicitPaddingConvolution.h"
#include "UnrolledInputImplicitInPaddingConvolution.h"
#include "UnrolledOutputConvolution.h"

void Test(const float* WFilMaj, const float* WRowMaj, const float* XRowMajExp, const float* XChlMajExp, const float* XRowMajImp, const float* XChlMajImp, int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int vStride, int hStride, int xPadTop, int xPadBottom, int xPadLeft, int xPadRight)
{
    // output shape
    int yChls = wCount;

    // input shape
    int xRows = (yRows - 1) * vStride + wRows; // includes any input padding
    int xCols = (yCols - 1) * hStride + wCols; // includes any input padding

    // allocate output tensors
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3Order);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3Order);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3Order);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3Order);

    // convolutions

    std::cout << "ForLoopConvolution" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        Convolution(properties, WFilMaj, XRowMajExp, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "UnrolledInputConvolution" << std::endl;
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj, XRowMajExp, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "UnrolledInputChlMajInputConvolution" << std::endl;
    if(hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
        Convolution(properties, WFilMaj, XChlMajExp, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "UnrolledOutputConvolution" << std::endl;
    {
        auto properties = ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        auto space = std::vector<float>(xRows * xCols * wCount * wRows * wCols);
        Convolution(properties, WFilMaj, XRowMajExp, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        std::cout << YChlMaj << std::endl << std::endl;
    }

    std::cout << "UnrolledInputImplicitInPaddingConvolution" << std::endl;
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>(9 * wChls * yRows * yCols);
        Convolution(properties, WFilMaj, XChlMajImp, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "UnrolledInputExplicitOutPaddingConvolution" << std::endl;
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj, XChlMajExp, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    std::cout << "UnrolledInputExplicitPaddingConvolution" << std::endl;
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        Convolution(properties, WFilMaj, XChlMajExp, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    // partially unrolled convolutions
    std::cout << "PartiallyUnrolledInputImplicitInPaddingConvolution" << std::endl;
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        auto space = std::vector<float>(yRows * yCols * wChls);
        Convolution(properties, WRowMaj, XRowMajImp, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        std::cout << YRowMaj << std::endl << std::endl;
    }

    std::cout << "PartiallyUnrolledInputExplicitOutPaddingConvolution" << std::endl;
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj, XRowMajExp, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        std::cout << YRowMajExp << std::endl << std::endl;
    }

    std::cout << "PartiallyUnrolledInputExplicitPaddingConvolution" << std::endl;
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj, XRowMajExp, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        std::cout << YRowMajExp << std::endl << std::endl;
    }
}

void Test(int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int vStride, int hStride)
{
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

    Test(WFilMaj.Data(), WRowMaj.Data(), XRowMajExp.Data(), XChlMajExp.Data(), XRowMajImp.Data(), XChlMajImp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, vStride, hStride, xPadTop, xPadBottom, xPadLeft, xPadRight);    
}

void Test()
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

    Test(WFilMaj.Data(), WRowMaj.Data(), XRowMajExp.Data(), XChlMajExp.Data(), XRowMajImp.Data(), XChlMajImp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, vStride, hStride, xPadTop, xPadBottom, xPadLeft, xPadRight);    
}

void Benchmark(double testDuration, int xCount, int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int vStride, int hStride)
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

    // ForLoopConvolution
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        });
        std::cout << time << ", ";
    }

    // UnrolledInputConvolution
    {
        auto properties = ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }

    // UnrolledInputChlMajInputConvolution
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

    // UnrolledOutputConvolution
    {
        auto properties = ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        auto space = std::vector<float>(xRows * xCols * wCount * wRows * wCols);
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }

    // UnrolledInputImplicitInPaddingConvolution
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>(9 * wChls * yRows * yCols);
        auto time = RunBenchmark<float>(testDuration, XChlMajImp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputExplicitOutPaddingConvolution
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XChlMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputExplicitPaddingConvolution
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XChlMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputImplicitInPaddingConvolution
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        auto space = std::vector<float>(yRows * yCols * wChls);
        auto time = RunBenchmark<float>(testDuration, XRowMajImp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputExplicitOutPaddingConvolution
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputExplicitPaddingConvolution
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = RunBenchmark<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
        });
        std::cout << time << std::endl;
    }
    else
    {
        std::cout << "n/a" << std::endl;
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "usage: convolutional <benchmark.csv>\n";
        exit(1);
    }

    auto parser = CSVParser<int>(argv[1]);

    if(!parser.IsValid())
    {
        std::cout << "error openning and parsing file " << argv[1] << std::endl;
        exit(1);
    }
    
    std::vector<std::string> requiredKeys = {"wCount", "wRows", "wCols", "wChls", "yRows", "yCols", "vStride", "hStride"};
    if(!parser.HeaderContains(requiredKeys))
    {
        std::cout << argv[1] << " missing required columns\n";
        exit(1);
    }

    // print the output header
    for(auto key : requiredKeys)
    {
        std::cout << key << ", ";
    }

    std::cout << "ForLoopConvolution, ";
    std::cout << "UnrolledInputConvolution, ";
    std::cout << "UnrolledInputChlMajInputConvolution, ";
    std::cout << "UnrolledOutputConvolution, ";
    std::cout << "UnrolledInputImplicitInPaddingConvolution, ";
    std::cout << "UnrolledInputExplicitOutPaddingConvolution, ";
    std::cout << "UnrolledInputExplicitPaddingConvolution, ";
    std::cout << "PartiallyUnrolledInputImplicitInPaddingConvolution, ";
    std::cout << "PartiallyUnrolledInputExplicitOutPaddingConvolution, ";
    std::cout << "PartiallyUnrolledInputExplicitPaddingConvolution";
    std::cout << std::endl;

    // run benchmarks
    double testDuration = 1000;
    int xCount = 10;

    try
    {
        while(parser.IsValid())
        {
            auto parameters = parser[requiredKeys];
            for(auto p : parameters)
            {
                std::cout << p << ", ";
            }
            
            Benchmark(testDuration, xCount, parser["wCount"], parser["wRows"], parser["wCols"], parser["wChls"], parser["yRows"], parser["yCols"], parser["vStride"], parser["hStride"]);
            parser.Next();
        }
    }
    catch(ParserException e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

