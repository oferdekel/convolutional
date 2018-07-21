////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     PaddedConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "Tensor.h"

template <typename ElementType>
void PaddedConvolution(const ElementType* WChlMaj, const ElementType* XChlMajImp, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    assert(hStride == 1);
    assert(vStride == 1);
    assert(wRows == 3);
    assert(wCols == 3);

    int xRows = yRows + 2;
    int xCols = yCols + 2;

    int uRows = yRows * yCols;
    int uCols = 9 * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = WRowMaj;
    ElementType* ZRowMaj = YRowMaj;

    std::vector<ElementType> UColMaj(uRows * uCols);

    // unroll input
    for(int wChl = 0; wChl < wChls; ++wChl)
    {
        
    }

    // matrix-matrix multiply
    Gemm(false, false, true, uRows, vCols, uCols, 1, UColMaj.data(), VColMaj, 1, ZRowMaj);
}
