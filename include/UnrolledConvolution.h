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

// Unrolled-input convolution with row-major input tensor and row-major output tensor 
//
// W - 4-dimensional weights tensor in row-major order
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
//
template <typename ElementType>
void Convolution(ConvolutionProperties<RowMajorInput, RowMajorOutput, UnrolledInput>,
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
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y;

    // allocate a column-major matrix U to hold unrolled input
    std::vector<ElementType> URowMaj(uRows * uCols);
    int copySize = wCols * wChls;

    // unroll input
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
                float* target = URowMaj.data() + (uRow * wRows + wRow) * copySize;

                // copy from X to U
                std::copy(source, source + copySize, target);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(true, false, true, uRows, vCols, uCols, 1, URowMaj.data(), VColMaj, 0, ZRowMaj);
}

// Unrolled-input convolution with channel-major input tensor and row-major output tensor 
//
// W - 4-dimensional weights tensor in row-major order
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
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, RowMajorOutput, UnrolledInput>,
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
    if (hStride != 1)
    {
        throw std::invalid_argument("Unrolled Convolution requires hStride = 1");
    }

    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = yCols + wCols - 1;

    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y;

    // allocate a column-major matrix U to hold unrolled input
    std::vector<ElementType> UColMaj(uRows * uCols);
    int copySize = yCols;

    // unroll input
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
                    ElementType* target = UColMaj.data() + (uCol * yRows + yRow) * yCols;

                    // copy from X to U
                    std::copy(source, source + copySize, target);
                }   
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, UColMaj.data(), VColMaj, 0, ZRowMaj);
}

// Unrolled-output convolution with row-major input tensor and row-major output tensor 
//
// W - 4-dimensional weights tensor in row-major order
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
//
template <typename ElementType>
void Convolution(ConvolutionProperties<RowMajorInput, RowMajorOutput, UnrolledOutput>,
    const ElementType* W, const ElementType* X, ElementType* Y, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    throw std::invalid_argument("Not yet implemented");
}