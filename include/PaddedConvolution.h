////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PaddedConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "ConvolutionProperties.h"
#include "Tensor.h"

#include <string>
#include <exception>

// Structured delete subroutine
template <typename ElementType>
void StructuredDelete(ElementType* begin, int skip, int singles, int size,  int intervals)
{
    begin += skip;
    for(int i = 0; i < singles; ++i)
    {
        *begin = 0;
        begin += skip;
    }

    for(int j = 0; j < intervals; ++j)
    {
        std::fill_n(begin, size, (ElementType)0);
        begin += size + skip - 1;
        for(int i = 0; i < singles; ++i)
        {
            *begin = 0;
            begin += skip;
        } 
    }
}

// Unrolled-input convolution with implicit input padding, with 3x3 filters. Assumes a channel-major input tensor with an implicit row of zero-padding on the top/bottom and a column of zero-padding on the right/left, and a row-major output tensor.
//
// W - 4-dimensional weights tensor in filter-major order, which represents 3x3 filters 
// X - 3-dimensional input tensor in channel-major order with implicit zero-padding
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
void Convolution(ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, UnrolledInput>,
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

    // allocate a column-major matrix U to hold unrolled input
    int uRows = yRows * yCols;
    int uCols = 9 * wChls;
    std::vector<ElementType> UColMaj(uRows * uCols);
    ElementType* UColMajBlock = UColMaj.data();

    auto blockSize = yCols * yRows * wChls;

    // unroll input
    // input block corresponding to TOP LEFT filter elements (across all channels)
    std::copy(X, X + blockSize - yCols - 1, UColMajBlock + yCols + 1);
    StructuredDelete(UColMajBlock + yCols, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    // input block corresponding to TOP CENTER filter elements (across all channels)
    std::copy(X, X + blockSize - yCols, UColMajBlock + yCols);
    StructuredDelete(UColMajBlock + yCols - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    UColMajBlock += blockSize;

    // input block corresponding to TOP RIGHT filter elements (across all channels)
    std::copy(X + 1, X + blockSize - yCols, UColMajBlock + yCols);
    StructuredDelete(UColMajBlock + yCols - 1, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    // input block corresponding to MID LEFT filter elements (across all channels)
    std::copy(X, X + blockSize - 1, UColMajBlock + 1);
    StructuredDelete(UColMajBlock, yCols, yRows * wChls - 1, 0, 0);
    UColMajBlock += blockSize;

    // input block corresponding to MID CENTER filter elements (across all channels)
    std::copy(X, X + blockSize, UColMajBlock);
    UColMajBlock += blockSize;

    // input block corresponding to MID RIGHT filter elements (across all channels)
    std::copy(X + 1, X + blockSize, UColMajBlock);
    StructuredDelete(UColMajBlock - 1, yCols, yRows * wChls - 1, 0, 0);
    UColMajBlock += blockSize;

    // input block corresponding to BOTTOM LEFT filter elements (across all channels)
    std::copy(X + yCols, X + blockSize - 1, UColMajBlock + 1);
    StructuredDelete(UColMajBlock, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    // input block corresponding to BOTTOM CENTER filter elements (across all channels)
    std::copy(X + yCols, X + blockSize, UColMajBlock);
    StructuredDelete(UColMajBlock - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    UColMajBlock += blockSize;

    // input block corresponding to BOTTOM RIGHT filter elements (across all channels)
    std::copy(X + yCols + 1, X + blockSize, UColMajBlock);
    StructuredDelete(UColMajBlock - 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, UColMaj.data(), W, 0, Y);
}

// Unrolled-input convolution with implicit input padding, with channel-major input tensor and row-major output tensor 
//
// W - 4-dimensional weights tensor in filter-major order
// X - 3-dimensional input tensor in channel-major order
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
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, RowMajorOutput, UnrolledInput>, 
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

    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y + (xCols * yPadTop + yPadLeft) * wCount;

    std::vector<ElementType> UColMaj(uRows * uCols);
    int copySize = uRows;

    // unroll input
    for(int wRow = 0; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            for(int wChl = 0; wChl < wChls; ++wChl) 
            {
                // calculate copy source
                const float* source = X + (wChl * xRows + wRow) * xCols + wCol;

                // calculate copy target
                int uCol = (wRow * wCols + wCol) * wChls + wChl;
                float* target = UColMaj.data() + uCol * copySize;

                // copy from X to U
                std::copy(source, source + copySize, target);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, UColMaj.data(), W, 0, ZRowMaj);

    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = ZRowMaj + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}


int GetDistToContent(int xRow, int xCol, int xRows, int xCols, int xPadTop, int xPadLeft)
{
    if(xRow < xPadTop)
    {
        return (xPadTop - xRow) * xCols + xPadLeft - xCol;
    }
    if(xCol < xPadLeft)
    {
        return xPadLeft - xCol;
    }
    return 0;
}

int GetDistFromContent(int xRow, int xCol, int xRows, int xCols, int xPadBottom, int xPadRight)
{
    int firstBottomPadRow = xRows - xPadBottom;
    if(xRow >= firstBottomPadRow)
    {
        return (xRow - firstBottomPadRow) * xCols + xPadRight + xCol + 1;
    }
    int firstRightPadCol = xCols - xPadRight;
    if(xCol >= firstRightPadCol)
    {
        return xCol - firstRightPadCol + 1;
    }
    return 0;
}

// Unrolled-input convolution with implicit input padding, with channel-major input tensor and row-major output tensor 
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
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
//
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, RowMajorOutput, UnrolledInput>, 
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
    int xPadTop,
    int xPadLeft,
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

    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y + (xCols * yPadTop + yPadLeft) * wCount;

    std::vector<ElementType> UColMaj(uRows * uCols);
    int copySize = uRows;

    // unroll input
    for(int wRow = 0; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            // get distances
            int distToContent = GetDistToContent(wRow, wCol, xRows, xCols, xPadTop, xPadLeft);
            int distFromContent = GetDistFromContent(wRow, wCol, xRows, xCols, xPadTop, xPadLeft);

            for(int wChl = 0; wChl < wChls; ++wChl) 
            {
                // calculate copy source
                const float* source = X + (wChl * xRows + wRow) * xCols + wCol;

                // calculate copy target
                int uCol = (wRow * wCols + wCol) * wChls + wChl;
                float* target = UColMaj.data() + uCol * copySize;

                // copy from X to U
                std::copy(source + distToContent, source + copySize - distFromContent, target + distToContent);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, UColMaj.data(), W, 0, ZRowMaj);

    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = ZRowMaj + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}