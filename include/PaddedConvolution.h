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

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only 3x3 receptive field 
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order, with an implicit row/col of zero-padding on the top/bottom/left/right
// * output tensor in row-major order
// * requires temporary space of size (9 * wChls * yRows * yCols)
//
// W - 4-dimensional weights tensor in filter-major order, which represents 3x3 filters 
// X - 3-dimensional input tensor in channel-major order with implicit zero-padding
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least (9 * wChls * yRows * yCols)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, FilterMajorFilters, ImplicitInputPadding, RowMajorOutput, ThreeByThreeField, UnitStride, UnrolledInput>,
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wChls, 
    int yRows, 
    int yCols,
    ElementType* space)
{
    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols;
    int uCols = 9 * wChls;
    ElementType* U = space;

    auto blockSize = yCols * yRows * wChls;

    auto processFilterPosition = [&](int position, int xOffset, int xSizeOffset, int uOffset, int skip, int singles, int size, int intervals)
    {
        // copy input from X into U
        ElementType* ptr = U + position * blockSize + uOffset;
        std::copy(X + xOffset, X + blockSize + xOffset + xSizeOffset, ptr);
        
        // structured delete of unneeded elements
        ptr += skip - 1;
        for(int i = 0; i < singles; ++i)
        {
            *ptr = 0;
            ptr += skip;
        }

        for(int j = 0; j < intervals; ++j)
        {
            std::fill_n(ptr, size, (ElementType)0);
            ptr += size + skip - 1;
            for(int i = 0; i < singles; ++i)
            {
                *ptr = 0;
                ptr += skip;
            } 
        }
    }; 

    // unroll input block corresponding to TOP LEFT filter elements (across all channels)
    processFilterPosition(0, 0, -yCols - 1, yCols + 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to TOP CENTER filter elements (across all channels)
    processFilterPosition(1, 0, -yCols, yCols, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to TOP RIGHT filter elements (across all channels)
    processFilterPosition(2, 1, -yCols - 1, yCols, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to MID LEFT filter elements (across all channels)
    processFilterPosition(3, 0, -1, 1, yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to MID CENTER filter elements (across all channels)
    std::copy(X, X + blockSize, U + 4 * blockSize);

    // unroll input block corresponding to MID RIGHT filter elements (across all channels)
    processFilterPosition(5, 1, -1, 0,  yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to BOTTOM LEFT filter elements (across all channels)
    processFilterPosition(6, yCols, -yCols - 1, 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to BOTTOM CENTER filter elements (across all channels)
    processFilterPosition(7, yCols, -yCols, 0, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to BOTTOM RIGHT filter elements (across all channels)
    processFilterPosition(8, yCols + 1, -yCols - 1, 0,  yCols, yRows - 2, yCols + 1, wChls - 1);

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, U, W, 0, Y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only 3x3 receptive field 
// * supports only horizontal and vertical stride of 1
// * partially unrolled input 
// * filters in row-major order
// * input tensor in row-major order, with an implicit row/col of zero-padding on the top/bottom/left/right
// * output tensor in row-major order
// * requires temporary space of size (wChls * yRows * yCols)

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

// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in row-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// yPadTop - the number of implicit zero-padding rows at the top of the input
// yPadLeft - the number of implicit zero-padding columns at the left of the input
// space - pointer to temporary space of size at least (yRows * yCols * wChls)
template <typename ElementType>
void Convolution(ConvolutionProperties<ImplicitInputPadding, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, ThreeByThreeField, UnitStride>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wChls, 
    int yRows, 
    int yCols,
    ElementType* space)
{
    // use temp space to store the partial unrolled input matrix P in row-major order
    ElementType* P = space;

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires temporary space of size ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)

// W - 4-dimensional weights tensor in filter-major order
// X - 3-dimensional input tensor in channel-major order
// Y - 3-dimensional zero-padded output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// space - pointer to temporary space of size at least ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput,  UnitStride, UnrolledInput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int yRows, 
    int yCols, 
    ElementType* space)
{
    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;

    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y + (xCols * yPadTop + yPadLeft) * wCount;

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
                float* target = U + uCol * copySize;

                // copy from X to U
                std::copy(source, source + copySize, target);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, U, W, 0, ZRowMaj);

    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = ZRowMaj + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in row-major order
// * input tensor in row-major order
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space

// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in row-major order
// Y - 3-dimensional zero-padded output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
template <typename ElementType>
void Convolution(ConvolutionProperties<ExplicitOutputPadding, OddField, PartiallyUnrolledInput, RowMajorFilters, RowMajorInput, RowMajorOutput, UnitStride>, 
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

    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;

    int vCols = wCount;
    int vSize = wChls * wCount;

    // allocate P to hold the partially unrolled input
    int pRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int pCols = wChls;

    const ElementType* VColMaj = W;
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    int copySize = pRows;

    // unroll input
    const ElementType* P = X;
    const ElementType* V = W;
    Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, 0, Z);

    for(int wCol = 1; wCol < wCols; ++wCol) 
    {
        P = X + wCol * xChls;
        V += vSize;
        Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, 1, Z);
    }   

    for(int wRow = 1; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            P = X + (wRow * xCols + wCol) * xChls;
            V += vSize;
            Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, 1, Z);
        }   
    }   

    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Z + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in filter-major order
// * input tensor in channel-major order with any number of explicit padding rows on the top/bottom and explicit padding columns on the left/right
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires temporary space of size ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)

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

// W - 4-dimensional weights tensor in filter-major order
// X - 3-dimensional input tensor in channel-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// xPadTop - number of input padding rows on the top and bottom
// xPadLeft - number of input padding columns on the left and right
// space - pointer to temporary space of size at least ((yRows * yCols + (yRows - 1) * (wCols - 1)) * wRows * wCols * wChls)
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, FilterMajorFilters, OddField, RowMajorOutput, UnitStride, UnrolledInput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int yRows, 
    int yCols, 
    int xPadTop,
    int xPadLeft,
    ElementType* space)
{
    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;

    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    const ElementType* VColMaj = W;
    ElementType* ZRowMaj = Y + (xCols * yPadTop + yPadLeft) * wCount;

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
                float* target = U + uCol * copySize;

                // copy from X to U
                std::copy(source + distToContent, source + copySize - distFromContent, target + distToContent);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, U, W, 0, ZRowMaj);

    // delete the padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = ZRowMaj + (yCols + xCols * yRow) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D Tensor Convolution
// * supports only odd number of filter rows and columns
// * supports only horizontal and vertical stride of 1
// * unrolled input 
// * filters in row-major order
// * input tensor in row-major order with any number of explicit padding rows on the top/bottom and explicit padding columns on the left/right
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space 

// W - 4-dimensional weights tensor in row-major order
// X - 3-dimensional input tensor in channel-major order
// Y - 3-dimensional output tensor in row-major order
// wCount - number of filters in W
// wRows - number of rows in each filter in W
// wCols - number of columns in each filter in W
// wChls - number of channels in each filter in W
// yRows - number of rows in the output tensor Y
// yCols - number of columns in the output tensor Y
// xPadTop - the number of implicit zero-padding rows at the top of the input
// xPadLeft - the number of implicit zero-padding columns at the left of the input
template <typename ElementType>
void Convolution(ConvolutionProperties<ChannelMajorInput, ExplicitInputPadding, ExplicitOutputPadding, OddField, RowMajorFilters, RowMajorOutput, UnitStride, UnrolledInput>, 
    const ElementType* W, 
    const ElementType* X, 
    ElementType* Y, 
    int wCount, 
    int wRows, 
    int wCols, 
    int wChls, 
    int yRows, 
    int yCols, 
    int xPadTop, 
    int xPadLeft)
{
        throw std::invalid_argument("Not yet implemented");  
}