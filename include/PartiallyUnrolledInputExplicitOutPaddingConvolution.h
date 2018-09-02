////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PartiallyUnrolledInputExplicitOutPaddingConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once 

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "Tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * partially unrolled input 
// * filters in row-major order
// * input tensor in row-major order
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space
//
// W: 4-dimensional weights tensor in row-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional zero-padded output tensor in row-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
template <typename ElementType>
void Convolution(ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitHorizontalStride, UnitVerticalStride>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int yRows, 
    int yCols)
{
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;

    // reshape the relevant part of the output tensor Y into a row-major matrix Z
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int wRow, int wCol, ElementType beta)
    {
        // reshape the relevant part of the input tensor X into a row-major matrix P
        int pRows = yRows * yCols + (yRows - 1) * (wCols - 1);
        int pCols = wChls;
        const ElementType* P = X + (wRow * xCols + wCol) * xChls;

        // reshape the relevant part of the filter tensor W into a row-major matrix V
       int vCols = wCount;
       int vSize = wChls * wCount;
       const ElementType* V = W + (wRow * wCols + wCol) * vSize;

        // perform the matrix-matrix multiplication
        Gemm(RowMaj, RowMaj, RowMaj, pRows, vCols, pCols, 1, P, V, beta, Z);
    };

    // process the TOP LEFT filter position across all channels
    ProcessFilterPosition(0, 0, 0);

    // process the rest of the TOP filter rows
    for(int wCol = 1; wCol < wCols; ++wCol) 
    {
        ProcessFilterPosition(0, wCol, 1);
    }

    // process the remaining filter rows
    for(int wRow = 1; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            ProcessFilterPosition(wRow, wCol, 1);
        }   
    }   

    // delete the values that were written into the output padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Z + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

