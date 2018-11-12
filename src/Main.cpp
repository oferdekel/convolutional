////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Main.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "BlasHelpers.h"
#include "ConvProperties.h"
#include "CSVParser.h"
#include "ForLoopConv.h"
#include "PartiallyUnrolledInputExplicitOutPaddingConv.h"
#include "PartiallyUnrolledInputExplicitPaddingConv.h"
#include "PartiallyUnrolledInputImplicitInPaddingConv.h"
#include "Tensor.h"
#include "TestHelpers.h"
#include "UnrolledInputConv_cI.h"
#include "UnrolledInputConv_rI.h"
#include "UnrolledInputExplicitOutPaddingConv.h"
#include "UnrolledInputExplicitPaddingConv.h"
#include "UnrolledInputImplicitInPaddingConv.h"
#include "UnrolledOutputConv.h"

void RunBenchmark(double testDuration, int xCount, int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int vStride, int hStride)
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
    auto XRowMajExp = GetRandomTensors<float, 3>(xCount, engine, { xRows, xCols, xChls }, RowMaj3, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XChlMajExp = GetRandomTensors<float, 3>(xCount, engine, { xRows, xCols, xChls }, ChlMaj3, {xPadTop, xPadLeft, 0}, {xPadBottom, xPadRight, 0});
    engine.seed(seed2);
    auto XRowMajImp = GetRandomTensors<float, 3>(xCount, engine, { yRows, yCols, xChls }, RowMaj3);
    engine.seed(seed2);
    auto XChlMajImp = GetRandomTensors<float, 3>(xCount, engine, { yRows, yCols, xChls }, ChlMaj3);

    // allocate output tensors
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3);

    // ForLoopConv
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
        });
        std::cout << time << ", ";
    }

    // UnrolledInputConv_rIrFrO
    {
        auto properties = ConvProperties<RowMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
            {
                Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }

    // UnrolledInputConv_rIrFcO
    {
        auto properties = ConvProperties<RowMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
            {
                Convolution(properties, WRowMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    // UnrolledInputConv_rIfFrO
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }

    // UnrolledInputConv_rIfFcO
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }

    // UnrolledInputConv_cIrFrO
    if(hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputConv_cIrFcO
    if(hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, RowMajorFilters, ChannelMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WRowMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputConv_cIfFrO
    if(hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputConv_cIfFcO
    if(hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, ChannelMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(wRows * wCols * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledOutputConv
    {
        auto properties = ConvProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        try
        {
            auto space = std::vector<float>(xRows * xCols * wCount * wRows * wCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }

    // UnrolledInputImplicitInPaddingConv
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>(9 * wChls * yRows * yCols);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajImp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputExplicitOutPaddingConv
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // UnrolledInputExplicitPaddingConv
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        try
        {
            auto space = std::vector<float>((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
            auto time = GetMeanExecutionTime<float>(testDuration, XChlMajExp, [&](const float* X)
            {
                Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputImplicitInPaddingConv
    if(wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        try
        {
            auto space = std::vector<float>(yRows * yCols * wChls);
            auto time = GetMeanExecutionTime<float>(testDuration, XRowMajImp, [&](const float* X)
            {
                Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
            });
            std::cout << time << ", ";
        }
        catch(...)
        {
            std::cout << "mem, ";
        }
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputExplicitOutPaddingConv
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
        {
            Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
        });
        std::cout << time << ", ";
    }
    else
    {
        std::cout << "n/a, ";
    }

    // PartiallyUnrolledInputExplicitPaddingConv
    if(vStride == 1 && hStride == 1)
    {
        auto properties = ConvProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>{};
        auto time = GetMeanExecutionTime<float>(testDuration, XRowMajExp, [&](const float* X)
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

#ifndef BLAS_VERSION
#define BLAS_VERSION "none"
#endif

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "usage: convolutional <benchmark.csv>\n";
        exit(1);
    }

    if(std::string(argv[1]) == "-i")
    {
        std::cout << "Blas version: " << BLAS_VERSION << std::endl; 
        exit(0);
    }

    auto parser = CSVParser<int>(argv[1]);

    if(!parser.IsValid())
    {
        std::cout << "error opening and parsing file " << argv[1] << std::endl;
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

    std::cout << "ForLoopConv, ";
    std::cout << "UnrolledInputConv_rIrFrO, ";
    std::cout << "UnrolledInputConv_rIrFcO, ";
    std::cout << "UnrolledInputConv_rIfFrO, ";
    std::cout << "UnrolledInputConv_rIfFcO, ";
    std::cout << "UnrolledInputConv_cIrFrO, ";
    std::cout << "UnrolledInputConv_cIrFcO, ";
    std::cout << "UnrolledInputConv_cIfFrO, ";
    std::cout << "UnrolledInputConv_cIfFcO, ";
    std::cout << "UnrolledOutputConv, ";
    std::cout << "UnrolledInputImplicitInPaddingConv, ";
    std::cout << "UnrolledInputExplicitOutPaddingConv, ";
    std::cout << "UnrolledInputExplicitPaddingConv, ";
    std::cout << "PartiallyUnrolledInputImplicitInPaddingConv, ";
    std::cout << "PartiallyUnrolledInputExplicitOutPaddingConv, ";
    std::cout << "PartiallyUnrolledInputExplicitPaddingConv";
    std::cout << std::endl;

    // run benchmarks
    double testDuration = 5000;
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
            
            try
            {
                RunBenchmark(testDuration, xCount, parser["wCount"], parser["wRows"], parser["wCols"], parser["wChls"], parser["yRows"], parser["yCols"], parser["vStride"], parser["hStride"]);
            }
            catch(...)
            {
                std::cout << "error\n";
            }
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

