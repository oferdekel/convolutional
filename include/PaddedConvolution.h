////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PaddedConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "Tensor.h"

#include <string>

template <typename ElementType>
void PaddedConvolutionDelete(ElementType* begin, int skip, int singles, int size,  int intervals)
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

template <typename ElementType>
void PaddedConvolution(const ElementType* WRowMaj, const ElementType* XChlMajImp, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    assert(hStride == 1);
    assert(vStride == 1);
    assert(wRows == 3);
    assert(wCols == 3);

    // allocate U matrix to hold unrolled input
    int uRows = yRows * yCols;
    int uCols = 9 * wChls;
    std::vector<ElementType> UColMaj(uRows * uCols);
    ElementType* UColMajBlock = UColMaj.data();

    auto blockSize = yCols * yRows * wChls;

    // unroll input
    std::copy(XChlMajImp, XChlMajImp + blockSize - yCols - 1, UColMajBlock + yCols + 1);
    PaddedConvolutionDelete(UColMajBlock + yCols, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp, XChlMajImp + blockSize - yCols, UColMajBlock + yCols);
    PaddedConvolutionDelete(UColMajBlock + yCols - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp + 1, XChlMajImp + blockSize - yCols, UColMajBlock + yCols);
    PaddedConvolutionDelete(UColMajBlock + yCols - 1, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp, XChlMajImp + blockSize - 1, UColMajBlock + 1);
    PaddedConvolutionDelete(UColMajBlock, yCols, yRows * wChls - 1, 0, 0);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp, XChlMajImp + blockSize, UColMajBlock);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp + 1, XChlMajImp + blockSize, UColMajBlock);
    PaddedConvolutionDelete(UColMajBlock - 1, yCols, yRows * wChls - 1, 0, 0);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp + yCols, XChlMajImp + blockSize - 1, UColMajBlock + 1);
    PaddedConvolutionDelete(UColMajBlock, yCols, yRows - 2, yCols + 1, wChls - 1);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp + yCols, XChlMajImp + blockSize, UColMajBlock);
    PaddedConvolutionDelete(UColMajBlock - 1, yCols * (yRows - 1) + 1, 0, yCols, wChls - 1);
    UColMajBlock += blockSize;

    std::copy(XChlMajImp + yCols + 1, XChlMajImp + blockSize, UColMajBlock);
    PaddedConvolutionDelete(UColMajBlock - 1, yCols, yRows - 2, yCols + 1, wChls - 1);

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, wCount, uCols, 1, UColMaj.data(), WRowMaj, 1, YRowMaj);
}
