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
struct None{};
struct UnrolledInput{};
struct PartiallyUnrolledInput{};
struct UnrolledOutput{};
struct RowMajorInput{};
struct ChannelMajorInput{};
struct RowMajorOutput{};
struct ChannelMajorOutput{};
struct ImplicitInputPadding{};
struct ExplicitInputPadding{};
struct ExplicitOutputPadding{};

// convenient way of collecting an arbitrary number of properties
template<typename ... T>
using ConvolutionProperties = std::tuple<T...>;
