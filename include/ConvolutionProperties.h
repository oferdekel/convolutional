////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ConvolutionProperties.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <tuple>

// properties used to specialize the implementation of convolution 
struct ChannelMajorInput{};
struct ChannelMajorOutput{};
struct ExplicitInputPadding{};
struct ExplicitOutputPadding{};
struct FilterMajorFilters{};
struct ImplicitInputPadding{};
struct OddField{};              // number of filter rows must be odd, number of filter columns must be odd
struct PartiallyUnrolledInput{};
struct RowMajorFilters{};
struct RowMajorInput{};
struct RowMajorOutput{};
struct ThreeByThreeField{};     // number of filter rows and columns equals three
struct UnitHorizontalStride{};
struct UnitVerticalStride{};
struct UnrolledInput{};
struct UnrolledOutput{};

// convenient way of collecting an arbitrary number of properties
template<typename ... T>
using ConvolutionProperties = std::tuple<T...>;
