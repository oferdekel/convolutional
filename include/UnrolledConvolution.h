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
// hStride - horizontal stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least (wRows * wCols * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, RowMajorOutput, UnrolledInput>,
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
    if (hStride != 1)
    {
        throw std::invalid_argument("Unrolled Convolution requires hStride = 1");
    }

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
// * unrolled output 
// * filters in filter-major order
// * input tensor in row-major order
// * output tensor in row-major order
// * requires temporary space of size TODO
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
// space - TODO
//
template <typename ElementType>
void Convolution(ConvolutionProperties<FilterMajorFilters, RowMajorInput, RowMajorOutput, UnrolledOutput>,
    const ElementType* W, const ElementType* X, ElementType* Y, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    //throw std::invalid_argument("Not yet implemented");

    // TODO - use externally allocated memory

    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = yCols + wCols - 1;

    int uRows = xRows * xCols;
    int uCols = wChls;
    int vCols = wCount * wRows * wCols;

    // matrix-matrix multiply
    std::vector<ElementType> UColMaj(uRows * vCols);
    Gemm(true, false, false, uRows, vCols, uCols, 1, X, W, 1, UColMaj.data());

    int uBlockRow = 1;
    int uBlockCol = 2;


    // collect values from the unrolled output
    for(int yRow = 0; yRow < yRows; ++yRow) {


    for(int wRow = 0; wRow < wRows; ++wRow) {
        for(int wCol = 0; wCol < wCols; ++wCol) {
            for(int wChl = 0; wChl < wChls; ++wChl) {

                    // calculate copy source
                    int xRow = yRow * vStride + wRow;
                    int xCol = wCol;
                    int xChl = wChl;
                    const float* source = X + (xChl * xRows + xRow) * xCols + xCol;
                    
                    // calculate copy target
                    int uCol =  (wRow * wCols + wCol) * wChls + wChl;
                    ElementType* target = UColMaj.data() + (uCol * yRows + yRow) * yCols;

                    // copy from X to U
                  //  std::copy(source, source + copySize, target);
                }   
            }  
        }   
    }   




    //MatrixConstInterface<float> T(ZColMaj.data(), {uRows, vCols}, ColMaj2Order);
    //std::cout << T << std::endl;

}