////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledInputConv_rI.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "BlasHelpers.h"
#include "ConvProperties.h"
#include "Tensor.h"

#include <cassert>

// Helper function that unrolls a channel-major input tensor into an unrolled input matrix
template <typename ElementType>
void RowMajInputUnroll(const ElementType* X, 
    ElementType* U,
    int wRows, 
    int wCols, 
    int wChls, 
    int vStride, 
    int hStride, 
    int yRows, 
    int yCols,
    int uRows,
    int uCols)
{
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    int copySize = wCols * wChls;

    for(int yRow = 0; yRow < yRows; ++yRow) 
    {
        for(int yCol = 0; yCol < yCols; ++yCol) 
        {
            for(int wRow = 0; wRow < wRows; ++wRow) 
            {
                // calculate copy source
                int xRow = yRow * vStride + wRow;
                int xCol = yCol * hStride;
                const float* source = X + (xRow * xCols + xCol) * xChls;

                // calculate copy target
                int uRow = yRow * yCols + yCol;
                float* target = U + (uRow * wRows + wRow) * copySize;

                // copy from X to U
                assert(source >= X);
                assert(source + copySize <= X + xRows * xCols * xChls);
                assert(target >= U);
                assert(target + copySize <= U + uRows * uCols);
                std::copy(source, source + copySize, target);
            }  
        }   
    }   
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * unrolled input 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in row-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
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
// space: pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>,
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
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in row-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the row-major input
    RowMajInputUnroll(X, U, wRows, wCols, wChls, vStride, hStride, yRows, yCols, uRows, uCols);

    // reshape the filters tensor W into a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(RowMaj, ColMaj, RowMaj, uRows, vCols, uCols, 1, U, V, 0, Z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * unrolled input 
// * filters in row-major order
// * input tensor in row-major order
// * output tensor in row-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
//
// W: 4-dimensional weights tensor in row-major order
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
// space: pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvProperties<RowMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>,
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
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in row-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the row-major input
    RowMajInputUnroll(X, U, wRows, wCols, wChls, vStride, hStride, yRows, yCols, uRows, uCols);

    // reshape the filters tensor W into a row-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(RowMaj, RowMaj, RowMaj, uRows, vCols, uCols, 1, U, V, 0, Z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * unrolled input 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in channel-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
//
// W: 4-dimensional weights tensor in filter-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional output tensor in channel-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// vStride: vertical stride
// hStride: horizontal stride
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
// space: pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvProperties<FilterMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>,
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
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in row-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the row-major input
    RowMajInputUnroll(X, U, wRows, wCols, wChls, vStride, hStride, yRows, yCols, uRows, uCols);

    // reshape the filters tensor W into a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a column-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(RowMaj, ColMaj, ColMaj, uRows, vCols, uCols, 1, U, V, 0, Z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * unrolled input 
// * filters in row-major order
// * input tensor in row-major order
// * output tensor in channel-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
//
// W: 4-dimensional weights tensor in row-major order
// X: 3-dimensional input tensor in row-major order
// Y: 3-dimensional output tensor in channel-major order
// wCount: number of filters in W
// wRows: number of rows in each filter in W
// wCols: number of columns in each filter in W
// wChls: number of channels in each filter in W
// vStride: vertical stride
// hStride: horizontal stride
// yRows: number of rows in the output tensor Y
// yCols: number of columns in the output tensor Y
// space: pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvProperties<RowMajorFilters, RowMajorInput, ChannelMajorOutput, UnrolledInput>,
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
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in row-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the row-major input
    RowMajInputUnroll(X, U, wRows, wCols, wChls, vStride, hStride, yRows, yCols, uRows, uCols);

    // reshape the filters tensor W into a row-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a column-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(RowMaj, RowMaj, ColMaj, uRows, vCols, uCols, 1, U, V, 0, Z);
}
