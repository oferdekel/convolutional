////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "Tensor.h"

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * unrolled input 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in row-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
//
// W - 4-dimensional weights tensor in filter-major order
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
// space - pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledInput>,
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
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    // use temp space to store the unrolled input matrix U in row-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the input
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
                std::copy(source, source + copySize, target);
            }  
        }   
    }   

    // reshape the filters tensor W into a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(true, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only horizontal stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order
// * output tensor in row-major order
// * requires temporary space of size (wRows * wCols * wChls * yRows * yCols)
//
// W - 4-dimensional weights tensor in filter-major order
// X - 3-dimensional input tensor in channel-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// vStride - vertical stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnitHorizontalStride, UnrolledInput>,
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int vStride, 
    int yRows, 
    int yCols,
    ElementType* space)
{
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = yCols + wCols - 1;

    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll the input
    int copySize = yCols;
    for(int wRow = 0; wRow < wRows; ++wRow) {
        for(int wCol = 0; wCol < wCols; ++wCol) {
            for(int wChl = 0; wChl < wChls; ++wChl) {
                for(int yRow = 0; yRow < yRows; ++yRow) {

                    // calculate copy source
                    int xRow = yRow * vStride + wRow;
                    int xCol = wCol;
                    int xChl = wChl;
                    const float* source = X + (xChl * xRows + xRow) * xCols + xCol;
                    
                    // calculate copy target
                    int uCol =  (wRow * wCols + wCol) * wChls + wChl;
                    ElementType* target = U + (uCol * yRows + yRow) * yCols;

                    // copy from X to U
                    std::copy(source, source + copySize, target);
                }   
            }  
        }   
    }   

    // reshape the filters tensor W into a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;
    
    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);
}

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
    int xCols = yCols + wCols - 1;

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
    Gemm(true, false, false, uRows, vCols, uCols, 1, U, V, 0, O);

    auto MultiVectorAdd = [&](ElementType* begin, int size, int count, int increment)
    {
        for(int i=0; i < count-1; ++i)
        {
            Axpy(size, begin + i * increment, begin + (i + 1) * increment);
            std::fill_n(begin + i * increment, size, (ElementType)0);
        }
    };

    int size = yCols;
    int count = wCols;
    int increment = uRows + hStride;

    // collect values from the unrolled output
    for(int filter = 0; filter < wCount; ++filter) {
        for(int yRow = 0; yRow < yRows; ++yRow) {

            int xRow = yRow * vStride;
        
            ElementType* first = O + filter * wRows * wCols * oRows + yRow * xCols;
            const ElementType* last = first + (count-1) * increment;

            MultiVectorAdd(first, size, count, increment);

            for(int wRow = 1; wRow < wRows; ++wRow) {

                int oFromRow = (xRow + wRow) * xCols;
                int oFromCol = (filter * wRows + wRow) * wCols;
                
                ElementType* next = O + oFromCol * oRows + oFromRow;
                Axpy(size, last, next);

                first = next;
                last = first + (count-1) * increment;
                MultiVectorAdd(first, size, count, increment);
            }

            ElementType* target = Y + (filter * yRows + yRow) * yCols;
            std::copy(last, last+size, target); 
        }   
    }   
}