////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"

template <typename ElementType>
void UnrollInput(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, size_t wCount, size_t wRows, size_t wCols, size_t wChls, size_t vStride, size_t hStride, size_t yRows, size_t yCols)
{

}

template <typename ElementType>
void UnrolledConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, size_t wCount, size_t wRows, size_t wCols, size_t wChls, size_t vStride, size_t hStride, size_t yRows, size_t yCols)
{
    size_t yChls = wCount;
    size_t xRows = (yRows - 1) * vStride + wRows;
    size_t xCols = (yCols - 1) * hStride + wCols;
    size_t xChls = wChls;

}
