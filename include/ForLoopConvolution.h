
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ForLoopConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ConvolutionProperties.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * straighforward for-loop implementation 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in row-major order
// * requires no temporary space 
//
//
// W: 4-dimensional weights tensor in filter-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional output tensor in row-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// vStride: vertical stride
// hStride: horizontal stride
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
template <typename ElementType>
void Convolution(ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput>,
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
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    for (int yRow = 0; yRow < yRows; ++yRow)
    {
        for (int yCol = 0; yCol < yCols; ++yCol)
        {
            for (int yChl = 0; yChl < yChls; ++yChl)
            {
                ElementType output = 0;
                for (int wRow = 0; wRow < wRows; ++wRow)
                {
                    for (int wCol = 0; wCol < wCols; ++wCol)
                    {
                        for (int wChl = 0; wChl < wChls; ++wChl)
                        {
                            auto weight = *(W + ((yChl * wRows + wRow) * wCols + wCol) * wChls + wChl);

                            auto xRow = yRow * vStride + wRow;
                            auto xCol = yCol * hStride + wCol;
                            auto xChl = wChl;
                            auto input = *(X + (xRow * xCols + xCol) * xChls + xChl);

                            output += weight * input;
                        }
                    }
                }

                *(Y + (yRow * yCols + yCol) * yChls + yChl) = output;
            }
        }
    }
}
