////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledInputExplicitPaddingConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once 

#include "BlasHelpers.h"
#include "ConvProperties.h"
#include "Tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order with any number of explicit padding rows on the top/bottom and explicit padding columns on the left/right
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires temporary space of size ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)
//
// W: 4-dimensional weights tensor in filter-major order
// X: 3-dimensional input tensor in channel-major order
// Y: 3-dimensional output tensor in row-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
// xPadTop: number of input padding rows on the top and bottom
// xPadLeft: number of input padding columns on the left and right
// space: pointer to temporary space of size at least ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)
template <typename ElementType>
void Convolution(ConvProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, UnrolledInput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int yRows, 
    int yCols, 
    int xPadTop,
    int xPadLeft,
    ElementType* space)
{
    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;
    int xPadBottom = xPadTop;
    int xPadRight = xPadLeft;

    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;
    std::fill_n(U, uRows * uCols, (ElementType)0);

    // unroll input
    int copySize = uRows;
    for(int wRow = 0; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            int xRow = wRow;
            int xCol = wCol;

            // get distances
            int distToContent = 0;
            if(xRow < xPadTop)
            {
                distToContent = (xPadTop - xRow) * xCols + xPadLeft - xCol;
            }
            else if(xCol < xPadLeft)
            {
                distToContent = xPadLeft - xCol;
            }

            int distFromContent = 0;
            if (wRows - xRow <= xPadBottom)
            {
                distFromContent = (xRow + xPadBottom - wRows + 1) * xCols + xCol + xPadRight - wCols + 1;
            }
            else if (wCols - xCol <= xPadRight)
            {
                distFromContent = xCol + xPadRight - wCols + 1;
            }

            for(int wChl = 0; wChl < wChls; ++wChl) 
            {
                // calculate copy source
                const float* source = X + (wChl * xRows + wRow) * xCols + wCol;

                // calculate copy target
                int uCol = (wRow * wCols + wCol) * wChls + wChl;
                float* target = U + uCol * copySize;

                // copy from X to U
                assert(source + distToContent >= X);
                assert(source + copySize - distFromContent <= X + xRows * xCols * xChls);
                assert(target + distToContent >= U);
                assert(target + copySize - distFromContent <= U + uRows * uCols);
                std::copy(source + distToContent, source + copySize - distFromContent, target + distToContent);
            }
        }   
    }   

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;

    // reshape the relevant part of the output tensor Y to a row-major matrix Z
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    // matrix-matrix multiply
    Gemm(ColMaj, ColMaj, RowMaj, uRows, vCols, uCols, 1, U, V, 0, Z);

    // delete the values that were written into the output padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Z + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

