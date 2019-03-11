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
#include "PartiallyUnrolledInputImplicitInPaddingConv.h"
#include "Tensor.h"
#include "TestHelpers.h"
#include "UnrolledInputConv_cI.h"
#include "UnrolledInputConv_rI.h"
#include "UnrolledInputExplicitOutPaddingConv.h"
#include "UnrolledInputExplicitPaddingConv.h"
#include "UnrolledInputImplicitInPaddingConv.h"
#include "UnrolledOutputConv.h"
#include "VirtuallyUnrolledInputExplicitOutPaddingConv.h"
#include "VirtuallyUnrolledInputExplicitPaddingConv.h"

template <typename TensorType>
void PrintBenchmark(bool condition, double testDuration, const std::vector<TensorType>& inputs, const BenchmarkType<float>& benchmark)
{
    if(!condition)
    {
        std::cout << "n/a";
        return;
    }

    try
    {
        auto time = GetMeanExecutionTime<float>(testDuration, inputs, benchmark);
        std::cout << time;
    }
    catch(...)
    {
        std::cout << "err";
    }

    std::cout.flush();
}

void RunAllBenchmarks(double testDuration, int xCount, int wCount, int wRows, int wCols, int wChls, int yRows, int yCols, int vStride, int hStride)
{
    // comparison tolerance (only in Debug compile)
    const double tolerance = 1.0e-3;

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
    auto YRef = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3);
    auto YRowMaj = Tensor<float,3>({ yRows, yCols, yChls }, RowMaj3);
    auto YRowMajExp = Tensor<float,3>({ xRows, xCols, yChls }, RowMaj3);
    auto YChlMaj = Tensor<float,3>({ yRows, yCols, yChls }, ChlMaj3);
    auto YChlMajExp = Tensor<float,3>({ xRows, xCols, yChls }, ChlMaj3);

    // scratch space
    std::vector<float> space;

    // bool to indicate whether to run and validate a benchmark
    bool benchCondition = true;

    // ForLoopConv
    benchCondition = true;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>{};
        Convolution(properties, WFilMaj.Data(), X, YRef.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    });
    std::cout << ", ";

    // UnrolledInputConv_rIrFrO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = true;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<RowMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_rIrFcO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = true;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<RowMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YChlMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_rIfFrO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = true;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_rIfFcO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = true;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<FilterMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YChlMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_cIrFrO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_cIrFcO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, RowMajorFilters, ChannelMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YChlMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_cIfFrO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputConv_cIfFcO
    space.resize(wRows * wCols * wChls * yRows * yCols);
    benchCondition = hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, ChannelMajorOutput, UnitHorizontalStride, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YChlMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledOutputConv
    space.resize(xRows * xCols * wCount * wRows * wCols);
    benchCondition = hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>{};
        Convolution(properties, WFilMaj.Data(), X, YChlMaj.Data(), wCount, wRows, wCols, wChls, vStride, hStride, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YChlMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputImplicitInPaddingConv
    space.resize(9 * wChls * yRows * yCols);
    benchCondition = wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajImp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // UnrolledInputExplicitOutPaddingConv
    space.resize((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
    benchCondition = vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMajExp.GetSubTensor({1,1,0}, YRef.Shape()), tolerance));
    }
    std::cout << ", ";

    // UnrolledInputExplicitPaddingConv
    space.resize((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls);
    benchCondition = vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XChlMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>{};
        Convolution(properties, WFilMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMajExp.GetSubTensor({1,1,0}, YRef.Shape()), tolerance));
    }
    std::cout << ", ";

    // PartiallyUnrolledInputImplicitInPaddingConv
    space.resize(yRows * yCols * wChls);
    benchCondition = wRows == 3 && wCols == 3 && vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XRowMajImp, [&](const float* X)
    {
        auto properties = ConvProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>{};
        Convolution(properties, WRowMaj.Data(), X, YRowMaj.Data(), wCount, wChls, yRows, yCols, space.data());
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMaj, tolerance));
    }
    std::cout << ", ";

    // VirtuallyUnrolledInputExplicitOutPaddingConv
    benchCondition = vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<ExplicitOutputPadding, OddField, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, VirtuallyUnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols);
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMajExp.GetSubTensor({1,1,0}, YRef.Shape()), tolerance));
    }
    std::cout << ", ";

    // VirtuallyUnrolledInputExplicitPaddingConv
    benchCondition = vStride == 1 && hStride == 1;
    PrintBenchmark(benchCondition, testDuration, XRowMajExp, [&](const float* X)
    {
        auto properties = ConvProperties<RowMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, VirtuallyUnrolledInput>{};
        Convolution(properties, WRowMaj.Data(), X, YRowMajExp.Data(), wCount, wRows, wCols, wChls, yRows, yCols, xPadTop, xPadLeft);
    });
    if (benchCondition)
    {
        assert(YRef.ApproxEquals(YRowMajExp.GetSubTensor({1,1,0}, YRef.Shape()), tolerance));
    }
    std::cout << std::endl;
}

void ProcessBenchmarksFile(CSVParser<int>& parser)
{
    std::vector<std::string> requiredKeys = {"wCount", "wRows", "wCols", "wChls", "yRows", "yCols", "vStride", "hStride"};
    if(!parser.HeaderContains(requiredKeys))
    {
        std::cerr << "file missing required columns\n";
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
    std::cout << "VirtuallyUnrolledInputExplicitOutPaddingConv, ";
    std::cout << "VirtuallyUnrolledInputExplicitPaddingConv";
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
            
            try
            {
                RunAllBenchmarks(testDuration, xCount, parser["wCount"], parser["wRows"], parser["wCols"], parser["wChls"], parser["yRows"], parser["yCols"], parser["vStride"], parser["hStride"]);
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
}

#ifndef BLAS_VERSION
#define BLAS_VERSION "none"
#endif

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "usage: convolutional <benchmark.csv> (or) convolutional -b\n";
        exit(1);
    }

    if(std::string(argv[1]) == "-b")
    {
        std::cout << "Blas version: " << BLAS_VERSION << std::endl; 
        exit(0);
    }

    #ifndef NDEBUG
    std::cout << "Warning: DEBUG BUILD" << std::endl;
    #endif 

    // create a parser for the benchmarks.csv file
    auto parser = CSVParser<int>(argv[1]);

    if(!parser.IsValid())
    {
        std::cout << "error opening and parsing file " << argv[1] << std::endl;
        exit(1);
    }
    
    ProcessBenchmarksFile(parser);

    return 0;
}

