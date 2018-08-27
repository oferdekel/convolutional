////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     TestHelpers.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Tensor.h"

#include <functional>

using TestType = std::function<Tensor<float,3>()>;

template <typename ElementType>
using BenchmarkType = std::function<void(const ElementType* X)>;

void RunTest(TestType test);

//
//
//

template <typename ElementType, typename TensorType>
double RunBenchmark(double testDuration, const std::vector<TensorType>& inputs, BenchmarkType<ElementType> benchmark)
{
    double time = 0.0;
    for(const auto& input : inputs)
    {
        benchmark(input.Data());
    }
    return time;
}