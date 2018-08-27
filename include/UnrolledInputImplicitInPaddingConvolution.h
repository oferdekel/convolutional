////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledInputImplicitInPaddingConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once 

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "Tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only 3x3 receptive field 
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order, with an implicit row/col of zero-padding on the top/bottom/left/right
// * output tensor in row-major order
// * requires temporary space of size (9 * wChls * yRows * yCols)
//
// W - 4-dimensional weights tensor in filter-major order, which represents 3x3 filters 
// X - 3-dimensional input tensor in channel-major order with implicit zero-padding
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least (9 * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>,
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wChls, 
    int yRows, 
    int yCols,
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols;
    int uCols = 9 * wChls;
    ElementType* U = space;
    std::fill_n(U, uRows * uCols, (ElementType)0);

    auto blockSize = yCols * yRows * wChls;

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int position, int xOffset, int xSizeOffset, int uOffset, int skip, int singles, int size, int intervals)
    {
        // copy input from X into U
        ElementType* ptr = U + position * blockSize + uOffset;
        std::copy(X + xOffset, X + blockSize + xOffset + xSizeOffset, ptr);
        
        // structured delete of unneeded elements
        ptr += skip - 1;
        for(int i = 0; i < singles; ++i)
        {
            *ptr = 0;
            ptr += skip;
        }

        for(int j = 0; j < intervals; ++j)
        {
            std::fill_n(ptr, size, (ElementType)0);
            ptr += size + skip - 1;
            for(int i = 0; i < singles; ++i)
            {
                *ptr = 0;
                ptr += skip;
            } 
        }
    }; 

    // unroll input block corresponding to TOP LEFT filter elements across all channels
    ProcessFilterPosition(0, 0, -yCols - 1, yCols + 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to TOP CENTER filter elements across all channels
    ProcessFilterPosition(1, 0, -yCols, yCols, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to TOP RIGHT filter elements across all channels
    ProcessFilterPosition(2, 1, -yCols - 1, yCols, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to MID LEFT filter elements across all channels
    ProcessFilterPosition(3, 0, -1, 1, yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to MID CENTER filter elements across all channels
    std::copy(X, X + blockSize, U + 4 * blockSize);

    // unroll input block corresponding to MID RIGHT filter elements across all channels
    ProcessFilterPosition(5, 1, -1, 0,  yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to BOTTOM LEFT filter elements across all channels
    ProcessFilterPosition(6, yCols, -yCols - 1, 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to BOTTOM CENTER filter elements across all channels
    ProcessFilterPosition(7, yCols, -yCols, 0, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to BOTTOM RIGHT filter elements across all channels
    ProcessFilterPosition(8, yCols + 1, -yCols - 1, 0,  yCols, yRows - 2, yCols + 1, wChls - 1);

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;

    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);
}

