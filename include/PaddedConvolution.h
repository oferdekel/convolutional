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

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int position, int xOffset, int xSizeOffset, int uOffset, int skip, int singles, int size, int intervals)
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

    // unroll input block corresponding to TOP LEFT filter elements across all channels
    ProcessFilterPosition(0, 0, -yCols - 1, yCols + 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to TOP CENTER filter elements across all channels
    ProcessFilterPosition(1, 0, -yCols, yCols, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to TOP RIGHT filter elements across all channels
    ProcessFilterPosition(2, 1, -yCols - 1, yCols, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to MID LEFT filter elements across all channels
    ProcessFilterPosition(3, 0, -1, 1, yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to MID CENTER filter elements across all channels
    std::copy(X, X + blockSize, U + 4 * blockSize);

    // unroll input block corresponding to MID RIGHT filter elements across all channels
    ProcessFilterPosition(5, 1, -1, 0,  yCols, yRows * wChls - 1, 0, 0);

    // unroll input block corresponding to BOTTOM LEFT filter elements across all channels
    ProcessFilterPosition(6, yCols, -yCols - 1, 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // unroll input block corresponding to BOTTOM CENTER filter elements across all channels
    ProcessFilterPosition(7, yCols, -yCols, 0, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);

    // unroll input block corresponding to BOTTOM RIGHT filter elements across all channels
    ProcessFilterPosition(8, yCols + 1, -yCols - 1, 0,  yCols, yRows - 2, yCols + 1, wChls - 1);

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;

    // reshape the output tensor Y into a row-major matrix Z
    ElementType* Z = Y;

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);
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
//
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
    int vCols = wCount;
    int vSize = wChls * wCount;

    auto MultiplyMatrices = [&](const ElementType* P, int pRows, int pCols, int position, ElementType beta, int yRow)
    {
        // reshape the relevant part of the filters tensor W into a row-major matrix V
        int vCols = wCount;
        int vSize = wChls * wCount;
        const ElementType* V = W + position * vSize;

        // reshape the relevant part of the output tensor Y into a row-major matrix Z
        ElementType* Z = Y + yRow * vCols;

        // perform matrix multiplication
        Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, beta, Z);
    };

    // define a helper function that handles a single spatial filter position without copying input data
    auto ProcessFilterPositionByReshape = [&](int position, int xRow, int xCol, int xContentRows, int yRow)
    {
        // reshape the relevant part of X to the partial unrolled-input matrix P
        int pRows = xContentRows;
        int pCols = wChls;
        const ElementType* P = X + (xRow * yCols + xCol) * wChls; 

        MultiplyMatrices(P, pRows, pCols, position, 1, yRow);
    };

    // define a helper function that handles a single spatial filter position by copying input data
    auto ProcessFilterPositionByCopy = [&](int position, ElementType beta, int xRow, int xCol, int xContentRows, int yRow)
    {
        // use temp space to store the partially unrolled input matrix P in row-major order
        int pRows = xContentRows;
        int pCols = wChls;
        ElementType* P = space;

        // copy the relevant part of X into P
        const ElementType* source = X + (xRow * yCols + xCol) * wChls; 
        int copySize = pRows * pCols;
        std::copy(source, source + copySize, P);

        // delete unwanted values from P
        for(int pRow = yCols - 1; pRow < pRows; pRow += yCols)
        {
            std::fill_n(P + pRow * pCols, pCols, (ElementType)0);
        }

        MultiplyMatrices(P, pRows, pCols, position, beta, yRow);
    };

    // process the TOP LEFT filter position across all channels
    ProcessFilterPositionByCopy(0, (ElementType)0, 0, 0, (yRows - 1) * yCols - 1, yCols + 1);

    // process the TOP CENTER filter position across all channels
    ProcessFilterPositionByReshape(1, 0, 0, (yRows - 1) * yCols, yCols);

    // process the TOP RIGHT filter position across all channels
    ProcessFilterPositionByCopy(2, (ElementType)1, 0, 1, (yRows - 1) * yCols - 1, yCols);

    // process the MID LEFT filter position across all channels
    ProcessFilterPositionByCopy(3, (ElementType)1, 0, 0, yRows * yCols - 1, 1);

    // process the MID CENTER filter position across all channels
    ProcessFilterPositionByReshape(4, 0, 0, yRows * yCols, 0);

    // process the MID RIGHT filter position across all channels
    ProcessFilterPositionByCopy(5, (ElementType)1, 0, 1, yRows * yCols - 1, 0);

    // process the BOTTOM LEFT filter position across all channels
    ProcessFilterPositionByCopy(6, (ElementType)1, 1, 0, (yRows - 1) * yCols - 1, 1);

    // process the BOTTOM CENTER filter position across all channels
    ProcessFilterPositionByReshape(7, 1, 0, (yRows - 1) * yCols, 0);

    // process the BOTTOM RIGHT filter position across all channels
    ProcessFilterPositionByCopy(8, (ElementType)1, 1, 1, (yRows - 1) * yCols - 1, 0);
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
//
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

    // unroll the input
    int copySize = uRows;
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

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;

    // reshape the relevant part of the output tensor Y to a row-major matrix Z
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    // perform the matrix-matrix multiplication
    Gemm(false, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);

    // delete the values that were written into the output padding
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
// * partially unrolled input 
// * filters in row-major order
// * input tensor in row-major order
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space
//
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
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;

    // reshape the relevant part of the output tensor Y into a row-major matrix Z
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int wRow, int wCol, ElementType beta)
    {
        // reshape the relevant part of the input tensor X into a row-major matrix P
        int pRows = yRows * yCols + (yRows - 1) * (wCols - 1);
        int pCols = wChls;
        const ElementType* P = X + (wRow * xCols + wCol) * xChls;

        // reshape the relevant part of the filter tensor W into a row-major matrix V
       int vCols = wCount;
       int vSize = wChls * wCount;
       const ElementType* V = W + (wRow * wCols + wCol) * vSize;

        // perform the matrix-matrix multiplication
        Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, beta, Z);
    };

    // process the TOP LEFT filter position across all channels
    ProcessFilterPosition(0, 0, 0);

    // process the rest of the TOP filter rows
    for(int wCol = 1; wCol < wCols; ++wCol) 
    {
        ProcessFilterPosition(0, wCol, 1);
    }

    // process the remaining filter rows
    for(int wRow = 1; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            ProcessFilterPosition(wRow, wCol, 1);
        }   
    }   

    // delete the values that were written into the output padding
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
//
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

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;
    int xPadBottom = xPadTop;
    int xPadRight = xPadLeft;

    // use temp space to store the unrolled input matrix U in column-major order
    int uRows = yRows * yCols + (yRows - 1) * (wCols - 1);
    int uCols = wRows * wCols * wChls;
    ElementType* U = space;

    // unroll input
    int copySize = uRows;
    for(int wRow = 0; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            int xRow = wRow;
            int xCol = wCol;

            // get distances
            int distToContent = 0;
            if(xRow < xPadTop)
            {
                distToContent = (xPadTop - xRow) * xCols + xPadLeft - xCol;
            }
            else if(xCol < xPadLeft)
            {
                distToContent = xPadLeft - xCol;
            }

            int distFromContent = 0;
            if (wRows - xRow <= xPadBottom)
            {
                distFromContent = (xRow + xPadBottom - wRows + 1) * xCols + xCol + xPadRight - wCols + 1;
            }
            else if (wCols - xCol <= xPadRight)
            {
                distFromContent = xCol + xPadRight - wCols + 1;
            }

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

    // reshape the filter-major filter tensor W to a column-major matrix V
    int vCols = wCount;
    const ElementType* V = W;

    // reshape the relevant part of the output tensor Y to a row-major matrix Z
    ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount;

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, U, V, 0, Z);

    // delete the values that were written into the output padding
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
// * filters in row-major order
// * input tensor in row-major order with any number of explicit padding rows on the top/bottom and explicit padding columns on the left/right
// * output tensor in row-major order with (wRows - 1)/2 explicit padding rows on the top/bottom and (wCols - 1)/2 explicit padding columns on the left/right
// * requires no temporary space 
//
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
    int xRows = yRows + wRows - 1;
    int xCols = yCols + wCols - 1;
    int xChls = wChls;

    int yPadTop = (wRows - 1) / 2;
    int yPadLeft = (wCols - 1) / 2;
    int xPadBottom = xPadTop;
    int xPadRight = xPadLeft;

    // define a helper function that handles a single spatial filter position (row, col)
    auto ProcessFilterPosition = [&](int wRow, int wCol, ElementType beta)
    {
        int xRow = wRow;
        int xCol = wCol;

        // get distance from (xRow, xCol) to the beginning of the non-zero content
        int distToContent = 0;
        if(xRow < xPadTop)
        {
            distToContent = (xPadTop - xRow) * xCols + xPadLeft - xCol;
        }
        else if(xCol < xPadLeft)
        {
            distToContent = xPadLeft - xCol;
        }
        
        // get distance from the beginning of the non-zero content to (xRow, xCol)
        int distFromContent = 0;
        if(wRows - xRow <= xPadBottom)
        {
            distFromContent = (xRow + xPadBottom - wRows + 1) * xCols + xCol + xPadRight - wCols + 1;
        }
        else if(wCols - xCol <= xPadRight)
        {
            distFromContent = xCol + xPadRight - wCols + 1;
        }

        // reshape the relevant part of the input tensor X into the row-major matrix P
        int pRows = yRows * yCols + (yRows - 1) * (wCols - 1) - (distToContent + distFromContent);
        int pCols = wChls;
        const ElementType* P = X + (wRow * xCols + wCol + distToContent) * xChls;

        // reshape the relevant part of the filter tensor W into the row-major matrix V
        int vCols = wCount;
        int vSize = wChls * wCount;
        const ElementType* V = W + (wRow * wCols + wCol) * vSize;

        // reshape the relevant part of the output tensor Y into a row-major matrix Z
        ElementType* Z = Y + (xCols * yPadTop + yPadLeft) * wCount + distToContent * vCols;
        
        // perform matrix multiplication
        Gemm(true, true, true, pRows, vCols, pCols, 1, P, V, beta, Z);
    };

    // process the TOP LEFT filter position across all channels
    ProcessFilterPosition(0, 0, 0);

    // process the rest of the TOP filter rows
    for(int wCol = 1; wCol < wCols; ++wCol) 
    {
        ProcessFilterPosition(0, wCol, 1);
    }

    // process the remaining filter rows
    for(int wRow = 1; wRow < wRows; ++wRow) 
    {
        for(int wCol = 0; wCol < wCols; ++wCol) 
        {
            ProcessFilterPosition(wRow, wCol, 1);
        }   
    }   

    // delete the values that were written into the output padding
    int deleteSize = (wCols - 1) * wCount;
    for(int yRow = 0; yRow < yRows - 1; ++yRow)
    {
        ElementType* begin = Y + (xCols * (yRow + yPadTop) + (yCols + yPadLeft)) * wCount;
        std::fill(begin, begin + deleteSize, (ElementType)0);
    }
}