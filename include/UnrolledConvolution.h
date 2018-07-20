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
void UnrollInputRowMaj(const ElementType* XRowMaj, ElementType* URowMaj, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
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
                float* target = URowMaj + (uRow * wRows + wRow) * copySize;
                
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
void UnrolledConvolutionRowMaj(const ElementType* WChlMaj, const ElementType* XRowMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
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
    UnrollInputRowMaj(XRowMaj, URowMaj.data(), wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    Gemm(true, false, true, uRows, vCols, uCols, 1, URowMaj.data(), VColMaj, 1, ZRowMaj);
}

template <typename ElementType>
void UnrollInputChlMaj(const ElementType* XChlMaj, ElementType* UColMaj, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    assert(hStride == 1);

    int xRows = vStride * (yRows - 1) + wRows;
    int xCols = yCols + wCols - 1;
    int copySize = yCols;

    for(int wRow = 0; wRow < wRows; ++wRow) {
        for(int wCol = 0; wCol < wCols; ++wCol) {
            for(int wChl = 0; wChl < wChls; ++wChl) {
                for(int yRow = 0; yRow < yRows; ++yRow) {

                    // calculate memcpy target
                    int uCol =  (wRow * wCols + wCol) * wChls + wChl;
                    ElementType* target = UColMaj + (uCol * yRows + yRow) * yCols;

                    // calculate memcpy source
                    int xRow = yRow * vStride + wRow;
                    int xCol = wCol;
                    int xChl = wChl;
                    const float* source = XChlMaj + (xChl * xRows + xRow) * xCols + xCol;
                    
                    // copy from X to U
                    memcpy(target, source, copySize * sizeof(ElementType));
                }   
            }  
        }   
    }   
}

template <typename ElementType>
void UnrolledConvolutionChlMaj(const ElementType* WChlMaj, const ElementType* XChlMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;

    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = WChlMaj;
    ElementType* ZRowMaj = YRowMaj;

    std::vector<ElementType> UColMaj(uRows * uCols);
    UnrollInputChlMaj(XChlMaj, UColMaj.data(), wRows, wCols, wChls, vStride, hStride, yRows, yCols);
    Gemm(false, false, true, uRows, vCols, uCols, 1, UColMaj.data(), VColMaj, 1, ZRowMaj);
}
