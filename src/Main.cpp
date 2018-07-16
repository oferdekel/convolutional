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
    size_t numFilterChannels = 2;
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

    // random seeds and engine
    std::seed_seq seed1 = {103, 311, 1283};
    std::seed_seq seed2 = {3929, 437, 859};
    std::default_random_engine engine;

    // generate random filters
    engine.seed(seed1);
    auto W = GetRandomTensor<float>(engine, { numFilters, numFilterRows, numFilterColumns, numFilterChannels }, RowMajor4TensorOrder);
 
    // generate the same random input in both orders and with explicit/implicit padding
    engine.seed(seed2);
    auto XRowMajorExplicit = GetRandomTensor<float>(engine, { numInputRows, numInputColumns, numInputChannels }, RowMajor3TensorOrder, {verticalInputPadding, horizontalInputPadding, 0});

    engine.seed(seed2);
    auto XChannelMajorExplicit = GetRandomTensor<float>(engine, { numInputRows, numInputColumns, numInputChannels }, ChannelMajor3TensorOrder, {verticalInputPadding, horizontalInputPadding, 0});

    engine.seed(seed2);
    auto XRowMajorImplicit = GetRandomTensor<float>(engine, { numInputContentRows, numInputContentColumns, numInputContentChannels }, RowMajor3TensorOrder);

    engine.seed(seed2);
    auto XChannelMajorImplicit = GetRandomTensor<float>(engine, { numInputContentRows, numInputContentColumns, numInputContentChannels }, ChannelMajor3TensorOrder);

    return 0;
}