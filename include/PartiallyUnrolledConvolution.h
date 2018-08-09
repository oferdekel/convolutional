////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PartiallyUnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ConvolutionProperties.h"

template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ImplicitInputPadding, PartiallyUnrolledInput, ChannelMajorOutput>, 
    const ElementType* WRowMaj, 
    const ElementType* XRowMaj, 
    ElementType* YRowMaj, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int vStride, 
    int hStride, 
    int yRows, 
    int yCols)
{
    throw std::invalid_argument("Not yet implemented");
}