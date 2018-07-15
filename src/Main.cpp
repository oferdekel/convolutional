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
#include "Matrix.h"
#include "Tensor.h"

int main(int argc, char** argv)
{
    // define parameters
    size_t numFilterRows = 3;
    size_t numFilterColumns = 3;
    size_t numFilterChannels = 4;
    size_t numFilters = 3;

    size_t numOutputRows = 4;
    size_t numOutputColumns = 4;
    size_t numOutputChannels = numFilters;

    size_t verticalStride = 1;
    size_t horizontalStride = 1;

    size_t verticalInputPadding = (numFilterRows - 1) / 2; // on each side (top and bottom)
    size_t horizontalInputPadding = (numFilterColumns - 1) / 2; // on each side (left and right)

    size_t numInputRows = (numOutputRows - 1) * verticalStride + numFilterRows; // includes any input padding
    size_t numInputColumns = (numOutputColumns - 1) * horizontalStride + numFilterColumns; // includes any input padding
    size_t numInputChannels = numFilterChannels;

    size_t numInputContentRows = numInputRows - 2 * verticalInputPadding; // excludes any input padding
    size_t numInputContentColumns = numInputColumns - 2 * horizontalInputPadding; // excludes any input padding
    size_t numInputContentChannels = numInputChannels;

    // two random seeds
    std::seed_seq seed1 = {103, 311, 1283};
    std::seed_seq seed2 = {3929, 437, 859};

    // generate the same random filters in both orders by repeating the same seed
    auto WRowMajor = GetRandomTensors<float>(numFilters, std::default_random_engine(seed1), numFilterRows, numFilterColumns, numFilterChannels, TensorOrder::RowMajor);
    auto WColumnMajor = GetRandomTensors<float>(numFilters, std::default_random_engine(seed1), numFilterRows, numFilterColumns, numFilterChannels, TensorOrder::ChannelMajor);

    // generate the same random input in both orders and with explicit/implicit padding
    auto XRowMajorExplicit = GetRandomTensor<float>(std::default_random_engine(seed2), numInputRows, numInputColumns, numInputChannels, TensorOrder::RowMajor, verticalInputPadding, horizontalInputPadding);
    auto XColumnMajorExplicit = GetRandomTensor<float>(std::default_random_engine(seed2), numInputRows, numInputColumns, numInputChannels, TensorOrder::ChannelMajor, verticalInputPadding, horizontalInputPadding);
    auto XRowMajorImplicit = GetRandomTensor<float>(std::default_random_engine(seed2), numInputContentRows, numInputContentColumns, numInputContentChannels, TensorOrder::RowMajor);
    auto XColumnMajorImplicit = GetRandomTensor<float>(std::default_random_engine(seed2), numInputContentRows, numInputContentColumns, numInputContentChannels, TensorOrder::ChannelMajor);

    return 0;
}