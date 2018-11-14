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
#include <chrono>

template <typename ElementType>
using BenchmarkType = std::function<void(const ElementType* X)>;

// Executes the benchmark multiple times and measures mean execution time
template <typename ElementType, typename TensorType>
double GetMeanExecutionTime(double testDuration, const std::vector<TensorType>& inputs, const BenchmarkType<ElementType>& benchmark);

//
//
//

template <typename ElementType, typename TensorType>
double GetMeanExecutionTime(double testDuration, const std::vector<TensorType>& inputs, const BenchmarkType<ElementType>& benchmark)
{
    // warm up the caches
    for(const auto& input : inputs)
    {
        benchmark(input.Data());
    }

    int repetitions = 0;
    int duration = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // repeat the test until the desired duration is reached
    while (duration < testDuration)
    {
        for(const auto& input : inputs)
        {
            benchmark(input.Data());
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        duration = (int)std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();

        ++repetitions;
    }

    // calculate the mean time (in ms) per input
    auto millisecondsPerInput = static_cast<double>(duration) / (inputs.size() * repetitions);
    return millisecondsPerInput;
}