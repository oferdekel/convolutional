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
struct PartiallyUnrolledInput{};
struct RowMajorFilters{};
struct RowMajorInput{};
struct RowMajorOutput{};
struct UnrolledInput{};
struct UnrolledOutput{};

// convenient way of collecting an arbitrary number of properties
template<typename ... T>
using ConvolutionProperties = std::tuple<T...>;
