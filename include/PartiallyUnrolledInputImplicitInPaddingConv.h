////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PartiallyUnrolledInputImplicitInPaddingConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "BlasHelpers.h"
#include "ConvProperties.h"
#include "Tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only 3x3 receptive field 
// * supports only horizontal and vertical stride of 1
// * partially unrolled input 
// * filters in row-major order
// * input tensor in row-major order, with an implicit row/col of zero-padding on the top/bottom/left/right
// * output tensor in row-major order
// * requires temporary space of size (wChls * yRows * yCols)
//
// W: 4-dimensional weights tensor in row-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional output tensor in row-major order
// wCount: number of filters in W
// wChls: number of channels in each filter in W
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
// yPadTop: the number of implicit zero-padding rows at the top of the input
// yPadLeft: the number of implicit zero-padding columns at the left of the input
// space: pointer to temporary space of size at least (yRows * yCols * wChls)
template <typename ElementType>
void Convolution(ConvProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitHorizontalStride, UnitVerticalStride>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wChls, 
    int yRows, 
    int yCols,
    ElementType* space)
{
    int yChls = wCount;

    int xRows = yRows + 2;
    int xCols = yCols + 2;
    int xChls = wChls;

    int vCols = wCount;
    int vSize = wChls * wCount;

    auto MultiplyMatrices = [&](const ElementType* P, int pRows, int pCols, int position, int yRow)
    {
        // reshape the relevant part of the filters tensor W into a row-major matrix V
        int vCols = wCount;
        int vSize = wChls * wCount;
        const ElementType* V = W + position * vSize;

        // reshape the relevant part of the output tensor Y into a row-major matrix Z
        ElementType* Z = Y + yRow * vCols;

        // perform matrix multiplication
        Gemm(RowMaj, RowMaj, RowMaj, pRows, vCols, pCols, 1, P, V, 1, Z);
    };

    // define a helper function that handles a single spatial filter position without copying input data
    auto ProcessFilterPositionByReshape = [&](int position, int xRow, int xCol, int xContentRows, int yRow)
    {
        // reshape the relevant part of X to the partial unrolled-input matrix P
        int pRows = xContentRows;
        int pCols = wChls;
        const ElementType* P = X + (xRow * yCols + xCol) * wChls; 

        MultiplyMatrices(P, pRows, pCols, position, yRow);
    };

    // define a helper function that handles a single spatial filter position by copying input data
    auto ProcessFilterPositionByCopy = [&](int position, int xRow, int xCol, int xContentRows, int yRow)
    {
        // use temp space to store the partially unrolled input matrix P in row-major order
        int pRows = xContentRows;
        int pCols = wChls;
        ElementType* P = space;

        // copy the relevant part of X into P
        const ElementType* source = X + (xRow * yCols + xCol) * wChls; 
        int copySize = pRows * pCols;
        assert(source + copySize <= X + xRows * xCols * xChls);
        std::copy(source, source + copySize, P);

        // delete unwanted values from P
        for(int pRow = yCols - 1; pRow < pRows; pRow += yCols)
        {
            assert(pRow + 1 <= pRows);
            std::fill_n(P + pRow * pCols, pCols, (ElementType)0);
        }

        MultiplyMatrices(P, pRows, pCols, position, yRow);
    };

    // reset the output 
    std::fill_n(Y, yRows * yCols * yChls, (ElementType)0);

    // process the TOP LEFT filter position across all channels
    ProcessFilterPositionByCopy(0, 0, 0, (yRows - 1) * yCols - 1, yCols + 1);

    // process the TOP CENTER filter position across all channels
    ProcessFilterPositionByReshape(1, 0, 0, (yRows - 1) * yCols, yCols);

    // process the TOP RIGHT filter position across all channels
    ProcessFilterPositionByCopy(2, 0, 1, (yRows - 1) * yCols - 1, yCols);

    // process the MID LEFT filter position across all channels
    ProcessFilterPositionByCopy(3, 0, 0, yRows * yCols - 1, 1);

    // process the MID CENTER filter position across all channels
    ProcessFilterPositionByReshape(4, 0, 0, yRows * yCols, 0);

    // process the MID RIGHT filter position across all channels
    ProcessFilterPositionByCopy(5, 0, 1, yRows * yCols - 1, 0);

    // process the BOTTOM LEFT filter position across all channels
    ProcessFilterPositionByCopy(6, 1, 0, (yRows - 1) * yCols - 1, 1);

    // process the BOTTOM CENTER filter position across all channels
    ProcessFilterPositionByReshape(7, 1, 0, (yRows - 1) * yCols, 0);

    // process the BOTTOM RIGHT filter position across all channels
    ProcessFilterPositionByCopy(8, 1, 1, (yRows - 1) * yCols - 1, 0);
}

