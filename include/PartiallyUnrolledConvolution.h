////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PartiallyUnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ConvolutionProperties.h"

template <typename ElementType>
void ProcessFilterPosition(const ElementType* W, const ElementType* X, ElementType* P, ElementType* Y,  int wCount, int wChls,int yCols, int position, ElementType beta, int xRow, int xCol, int pRows, int yRow)
{
    int pCols = wChls;
    int vCols = wCount;
    int vSize = wChls * wCount;

    // copy the relevant part of X into the partial unrolled-input matrix P
    const ElementType* source = X + (xRow * yCols + xCol) * wChls; 
    int copySize = pRows * pCols;
    std::copy(source, source + copySize, P);

    // delete unwanted values from P
    for(int pRow = yCols - 1; pRow < pRows; pRow += yCols)
    {
        std::fill_n(P + pRow * pCols, pCols, (ElementType)0);
    }

    // define relevant submatrices of W and Y
    const ElementType* V = W + position * vSize;
    ElementType* Z = Y + yRow * vCols;

    // multiply
    Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, beta, Z);
}

template <typename ElementType>
void ProcessFilterPosition(const ElementType* W, const ElementType* X, ElementType* Y,  int wCount, int wChls, int yCols, int position, int xRow, int xCol, int pRows, int yRow)
{
    int pCols = wChls;
    int vSize = wChls * wCount;
    int vCols = wCount;

    // define relevant submatrices of X, W, and Y
    const ElementType* P = X + (xRow * yCols + xCol) * wChls; 
    const ElementType* V = W + position * vSize;
    ElementType* Z = Y + yRow * vCols;

    // multiply
    Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, 1, Z);
}

// Convolution with partially unrolled input, implicit input padding, and row-major input, output, and filter tensors 
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
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput>, 
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
    if (hStride != 1 || vStride != 1)
    {
        throw std::invalid_argument("Implicitly Padded Convolution requires hStride = 1 and vStride = 1");
    }
    if (wRows != 3 || wCols != 3)
    {
        throw std::invalid_argument("This implementation of Convolution is hard-coded for wRows = 3 and wCols = 3");
    }

    // allocate P to hold the partially unrolled input
    std::vector<ElementType> PRowMaj(yRows * yCols * wChls);
    ElementType* P = PRowMaj.data();

    // unroll input
    // process the TOP LEFT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 0, (ElementType)0, 0, 0, (yRows - 1) * yCols - 1, yCols + 1);

    // process the TOP CENTER filter position (across all channels)
    ProcessFilterPosition(W, X, Y, wCount, wChls, yCols, 1, 0, 0, (yRows - 1) * yCols, yCols);

    // process the TOP RIGHT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 2, (ElementType)1, 0, 1, (yRows - 1) * yCols - 1, yCols);

    // process the MID LEFT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 3, (ElementType)1, 0, 0, yRows * yCols - 1, 1);

    // process the MID CENTER filter position (across all channels)
    ProcessFilterPosition(W, X, Y, wCount, wChls, yCols, 4, 0, 0, yRows * yCols, 0);

    // process the MID RIGHT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 5, (ElementType)1, 0, 1, yRows * yCols - 1, 0);

    // process the BOTTOM LEFT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 6, (ElementType)1, 1, 0, (yRows - 1) * yCols - 1, 1);

    // process the BOTTOM CENTER filter position (across all channels)
    ProcessFilterPosition(W, X, Y, wCount, wChls, yCols, 7, 1, 0, (yRows - 1) * yCols, 0);

    // process the BOTTOM RIGHT filter position (across all channels)
    ProcessFilterPosition(W, X, P, Y, wCount, wChls, yCols, 8, (ElementType)1, 1, 1, (yRows - 1) * yCols - 1, 0);
}

// Convolution with partially unrolled input, explicit output padding, and row-major input, output, and filter tensors  
//
// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in row-major order
// Y - 3-dimensional zero-padded output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// vStride - vertical stride
// hStride - horizontal stride
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// yPadTop - the number of explicit zero-padding rows at the top of the output
// yPadLeft - the number of explicit zero-padding columns at the left of the output
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ExplicitOutputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput>, 
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
    int yPadTop, 
    int yPadLeft)
{
    if (hStride != 1 || vStride != 1)
    {
        throw std::invalid_argument("Implicitly Padded Convolution requires hStride = 1 and vStride = 1");
    }
    if (yPadTop * 2 + 1 != wRows || yPadLeft * 2 + 1 != wCols)
    {
        throw std::invalid_argument("yPapTop and yPadLeft must be consistent with wRows and wCols");
    }

    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    // allocate P to hold the partially unrolled input
    int pRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int pCols = wChls;

    const ElementType* VColMaj = W;
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    int copySize = pRows;

    // unroll input
    for(int wRow = 0; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            // partially unroll the input by reshaping
            
            const ElementType* P = X + (wRow * xCols + wCol) * wChls;

            // Gemm
        }   
    }   

                // const float* source = X + (wChl * xRows + wRow) * xCols + wCol;

                // // calculate copy target
                // int uCol = (wRow * wCols + wCol) * wChls + wChl;
                // float* target = UColMaj.data() + uCol * copySize;

                // // copy from X to U
                // std::copy(source, source + copySize, target);

    // for (int i = 0; i < _l; ++i)
    // {
    //     for (int j = 0; j < _m; ++j)
    //     {
    //         Gemm(CBLAS_ORDER::CblasRowMajor,
    //              CBLAS_TRANSPOSE::CblasNoTrans,
    //              CBLAS_TRANSPOSE::CblasTrans,
    //              _a * _b + (_a - 1) * _e,
    //              _c,
    //              _n,
    //              1,
    //              &X(i, j, 0),
    //              _n,
    //              _arrV.data() + (i * _m + j) * _n,
    //              _l * _m * _n,
    //              1,
    //              &_Y(_d / 2, _e / 2, 0),
    //              _c);
    //     }
    // }




    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Z + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

// Unrolled-input convolution with implicit input padding, with channel-major input tensor and row-major output tensor 
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
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, RowMajorOutput, UnrolledInput>, 
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
    int yPadTop, 
    int yPadLeft)
{
        throw std::invalid_argument("Not yet implemented");  
}