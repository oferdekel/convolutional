////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ConvProperties.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <tuple>

// properties used to specialize the implementation of convolution 
struct ChannelMajorInput{};     // input is provided in channel major tensor order
struct ChannelMajorOutput{};    // output is generated in channel major tensor order
struct ExplicitInputPadding{};  // input tensor includes explicit zero-padding
struct ExplicitOutputPadding{}; // output tensor includes explicit zero-padding
struct FilterMajorFilters{};    // filter tensor is given in filter, row, column, channel major-to-minor order
struct ImplicitInputPadding{};  // input should be processed with implicit zero-padding
struct OddField{};              // odd receptive field size - number of filter rows must be odd, number of filter columns must be odd
struct VirtuallyUnrolledInput{};// input is virtually unrolled piece by piece
struct RowMajorFilters{};       // filter tensor is given in row, column, channel, filter major-to-minor order
struct RowMajorInput{};         // input is provided in row major tensor order
struct RowMajorOutput{};        // output is provided in row major tensor order
struct ThreeByThreeField{};     // number of filter rows and columns must equal 3
struct UnitHorizontalStride{};  // horizontal stride must equal 1
struct UnitVerticalStride{};    // vertical stride must equal 1
struct UnrolledInput{};         // input is unrolled 
struct UnrolledOutput{};        // output is unrolled

// a convenient way of collecting an arbitrary number of properties in one type
template<typename ... T>
using ConvProperties = std::tuple<T...>;
