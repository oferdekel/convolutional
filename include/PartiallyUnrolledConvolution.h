////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PartiallyUnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ConvolutionProperties.h"

// Structured delete subroutine
template <typename ElementType>
void SpacedDelete(ElementType* begin, int size, int frequency, int count)
{
    begin += (frequency - 1) * size;
    for(int i = 0; i < count; ++i)
    {
        std::fill_n(begin, size, (ElementType)0);
        begin += frequency * size;
    }
}

// Convolution with partially unrolled input, implicit input padding, row-major input tensor, and row-major output tensor 
//
// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in row-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// vStride - vertical stride
// hStride - horizontal stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorInput, RowMajorOutput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int vStride, 
    int hStride, 
    int yRows, 
    int yCols)
{
    // allocate P to hold the partial unrolling input
    int pRows = yRows * yCols;
    int pCols = wChls;
    std::vector<ElementType> PRowMaj(pRows * pCols);

    // unroll input
    // input block corresponding to TOP LEFT filter elements (across all channels)
    // std::copy(X, X + blockSize - yCols - 1, URowMaj.data());
    // StructuredDelete(UColMajBlock + yCols, yCols, yRows - 2, yCols + 1, wChls - 1);
    // UColMajBlock += blockSize;

    // // input block corresponding to TOP CENTER filter elements (across all channels)
    // std::copy(X, X + blockSize - yCols, URowMaj.data());
    // StructuredDelete(UColMajBlock + yCols - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    // UColMajBlock += blockSize;

    // // input block corresponding to TOP RIGHT filter elements (across all channels)
    // std::copy(X + 1, X + blockSize - yCols, URowMaj.data());
    // StructuredDelete(UColMajBlock + yCols - 1, yCols, yRows - 2, yCols + 1, wChls - 1);
    // UColMajBlock += blockSize;

    // // input block corresponding to MID LEFT filter elements (across all channels)
    // std::copy(X, X + blockSize - 1, URowMaj.data());
    // StructuredDelete(UColMajBlock, yCols, yRows * wChls - 1, 0, 0);
    // UColMajBlock += blockSize;

    // // input block corresponding to MID CENTER filter elements (across all channels)
    // std::copy(X, X + blockSize, URowMaj.data());
    // UColMajBlock += blockSize;

    // // input block corresponding to MID RIGHT filter elements (across all channels)
    // std::copy(X + 1, X + blockSize, URowMaj.data());
    // StructuredDelete(UColMajBlock - 1, yCols, yRows * wChls - 1, 0, 0);
    // UColMajBlock += blockSize;

    // // input block corresponding to BOTTOM LEFT filter elements (across all channels)
    // std::copy(X + yCols, X + blockSize - 1, URowMaj.data());
    // StructuredDelete(UColMajBlock, yCols, yRows - 2, yCols + 1, wChls - 1);
    // UColMajBlock += blockSize;

    // // input block corresponding to BOTTOM CENTER filter elements (across all channels)
    // std::copy(X + yCols, X + blockSize, URowMaj.data());
    // StructuredDelete(UColMajBlock - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    // UColMajBlock += blockSize;

    // // input block corresponding to BOTTOM RIGHT filter elements (across all channels)
    // std::copy(X + yCols + 1, X + blockSize, URowMaj.data());
    // StructuredDelete(UColMajBlock - 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // // matrix-matrix multiply
    // Gemm(false, false, true, uRows, wCount, uCols, 1, UColMaj.data(), W, 1, Y);
}

template <typename ElementType>
void Convolution(ConvolutionProperties<ExplicitOutputPadding, PartiallyUnrolledInput, RowMajorInput, RowMajorOutput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
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

// Unrolled-input convolution with implicit input padding, with channel-major input tensor and row-major output tensor 
//
// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in channel-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// vStride - vertical stride
// hStride - horizontal stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, RowMajorOutput, UnrolledInput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int vStride, 
    int hStride, 
    int yRows, 
    int yCols, 
    int yPadTop, 
    int yPadLeft)
{
        throw std::invalid_argument("Not yet implemented");  
}