////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     VirtuallyUnrolledInputExplicitPaddingConv.h
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
// * partially unrolled input 
// * filters in row-major order
// * input tensor in row-major order with any number of explicit padding rows on the top/bottom and explicit padding columns on the left/right
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space 
//
// W: 4-dimensional weights tensor in row-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional output tensor in row-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
// xPadTop: the number of implicit zero-padding rows at the top of the input
// xPadLeft: the number of implicit zero-padding columns at the left of the input
template <typename ElementType>
void Convolution(ConvProperties<RowMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, RowMajorFilters, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride, VirtuallyUnrolledInput>, 
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
    int xPadLeft)
{
    int yChls = wCount;

    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;
    int xPadBottom = xPadTop;
    int xPadRight = xPadLeft;

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int wRow, int wCol)
    {
        int xRow = wRow;
        int xCol = wCol;

        // get distance from (xRow, xCol) to the beginning of the non-zero content
        int distToContent = 0;
        if(xRow < xPadTop)
        {
            distToContent = (xPadTop - xRow) * xCols + xPadLeft - xCol;
        }
        else if(xCol < xPadLeft)
        {
            distToContent = xPadLeft - xCol;
        }
        
        // get distance from the beginning of the non-zero content to (xRow, xCol)
        int distFromContent = 0;
        if(wRows - xRow <= xPadBottom)
        {
            distFromContent = (xRow + xPadBottom - wRows + 1) * xCols + xCol + xPadRight - wCols + 1;
        }
        else if(wCols - xCol <= xPadRight)
        {
            distFromContent = xCol + xPadRight - wCols + 1;
        }

        // reshape the relevant part of the input tensor X into the row-major matrix P
        int pRows = yRows * yCols + (yRows - 1) * (wCols - 1) - (distToContent + distFromContent);
        int pCols = wChls;
        const ElementType* P = X + (wRow * xCols + wCol + distToContent) * xChls;

        // reshape the relevant part of the filter tensor W into the row-major matrix V
        int vCols = wCount;
        int vSize = wChls * wCount;
        const ElementType* V = W + (wRow * wCols + wCol) * vSize;

        // reshape the relevant part of the output tensor Y into a row-major matrix Z
        ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount + distToContent * vCols;
        
        // perform matrix multiplication
        Gemm(RowMaj, RowMaj, RowMaj, pRows, vCols, pCols, 1, P, V, 1, Z);
    };

    // reset the output 
    assert((yRows + wRows - 1) * (yCols + wCols - 1) <= xRows * xCols);
    std::fill_n(Y, (yRows + wRows - 1) * (yCols + wCols - 1) * yChls, (ElementType)0);

    // process the TOP LEFT filter position across all channels
    ProcessFilterPosition(0, 0);

    // process the rest of the TOP filter rows
    for(int wCol = 1; wCol < wCols; ++wCol) 
    {
        ProcessFilterPosition(0, wCol);
    }

    // process the remaining filter rows
    for(int wRow = 1; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            ProcessFilterPosition(wRow, wCol);
        }   
    }   

    // delete the values that were written into the output padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Y + (xCols * (yRow + yPadTop) + (yCols + yPadLeft)) * wCount;
        assert(begin >= Y);
        assert(begin + deleteSize <= Y + xRows * xCols * yChls);
        std::fill_n(begin, deleteSize, (ElementType)0);
    }
}