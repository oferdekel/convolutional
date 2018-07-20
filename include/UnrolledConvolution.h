////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "Tensor.h"

#include <cstring> // memcpy
#include <vector> // std::vector

template <typename ElementType>
void UnrollInput(const ElementType* XRowMaj, ElementType* U, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int xCols = hStride * (yCols - 1) + wCols;
    int xChls = wChls;
    int copySize = wCols * wChls;
    
    for(int yRow = 0; yRow < yRows; ++yRow) 
    {
        for(int yCol = 0; yCol < yCols; ++yCol) 
        {
            for(int wRow = 0; wRow < wRows; ++wRow) 
            {
                // calculate memcpy target
                int uRow = yRow * yCols + yCol;
                float* target = U + (uRow * wRows + wRow) * copySize;
                
                // calculate memcpy source
                int xRow = yRow * vStride + wRow;
                int xCol = yCol * hStride;
                const float* source = XRowMaj + (xRow * xCols + xCol) * xChls;
                
                // copy from X to U
                memcpy(target, source, copySize * sizeof(ElementType));
            }  
        }   
    }   
}

template <typename ElementType>
void UnrolledConvolution(const ElementType* WChlMaj, const ElementType* XRowMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;

    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = WChlMaj;
    ElementType* ZRowMaj = YRowMaj;

    std::vector<ElementType> URowMaj(uRows * uCols);
    UnrollInput(XRowMaj, URowMaj.data(), wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    Gemm(true, false, true, uRows, vCols, uCols, 1, URowMaj.data(), VColMaj, 1, ZRowMaj);
}
