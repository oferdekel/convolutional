////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledOutputConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "Tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only horizontal stride of 1
// * unrolled output 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in channel-major order
// * requires temporary space of size (xRows * xCols * wCount * wRows * wCols)
//
// W - 4-dimensional weights tensor in filter-major order
// X - 3-dimensional input tensor in row-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// vStride - vertical stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least (xRows * xCols * wCount * wRows * wCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorOutput, FilterMajorFilters, RowMajorInput, UnrolledOutput>,
    const ElementType* W, const ElementType* X, ElementType* Y, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols, ElementType* space)
{
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;

    // reshape the row-major input tensor X to a row-major matrix U
    int uRows = xRows * xCols;
    int uCols = wChls;
    const ElementType* U = X;

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount * wRows * wCols;
    const ElementType* V = W;

    // use temp space to store the unrolled output matrix O in column-major order
    int oRows = uRows;    
    ElementType* O = space;
    Gemm(RowMaj, ColMaj, ColMaj, uRows, vCols, uCols, 1, U, V, 0, O);

    auto MultiVectorAdd = [&](ElementType* begin, int size, int count, int increment, int offset)
    {
        for(int i=0; i < count-1; ++i)
        {
            Axpy(size, 1, begin + i * offset, increment, begin + (i + 1) * offset, increment);
            std::fill_n(begin + i * offset, size, (ElementType)0); // TODO remove
        }
    };

    int size = yCols;
    int count = wCols;
    int offset = uRows + 1;

    // collect values from the unrolled output
    for(int filter = 0; filter < wCount; ++filter) {
        for(int yRow = 0; yRow < yRows; ++yRow) {

            int xRow = yRow * vStride;
        
            ElementType* first = O + filter * wRows * wCols * oRows + xRow * xCols;
            const ElementType* last = first + (count - 1) * offset;
            MultiVectorAdd(first, size, count, hStride, offset);

            for(int wRow = 1; wRow < wRows; ++wRow) {

                int oFromRow = (xRow + wRow) * xCols;
                int oFromCol = (filter * wRows + wRow) * wCols;
                
                ElementType* next = O + oFromCol * oRows + oFromRow;
                Axpy(size, 1, last, hStride, next, hStride);

                first = next;
                last = first + (count - 1) * offset;
                MultiVectorAdd(first, size, count, hStride, offset);
            }

            ElementType* target = Y + (filter * yRows + yRow) * yCols;
            Copy(size, last, hStride, target, 1); 
        }   
    }   
}