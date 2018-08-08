////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     UnrolledConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "Tensor.h"

#include <vector> // std::vector

template <typename ElementType>
void UnrolledConvolutionRowMaj(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    int uRows = yRows * yCols;
    int uCols = wRows * wCols * wChls;
    int vCols = wCount;

    const ElementType* VColMaj = WRowMaj;
    ElementType* ZRowMaj = YRowMaj;

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
                const float* source = XRowMaj + (xRow * xCols + xCol) * xChls;

                // calculate copy target
                int uRow = yRow * yCols + yCol;
                float* target = URowMaj.data() + (uRow * wRows + wRow) * copySize;

                // copy from X to U
                std::copy(source, source + copySize, target);
            }  
        }   
    }   

    // matrix-matrix multiply
    Gemm(true, false, true, uRows, vCols, uCols, 1, URowMaj.data(), VColMaj, 1, ZRowMaj);
}

template <typename ElementType>
void UnrolledConvolutionChlMaj(const ElementType* WRowMaj, const ElementType* XChlMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
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

    const ElementType* VColMaj = WRowMaj;
    ElementType* ZRowMaj = YRowMaj;

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
                    const float* source = XChlMaj + (xChl * xRows + xRow) * xCols + xCol;
                    
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
    Gemm(false, false, true, uRows, vCols, uCols, 1, UColMaj.data(), VColMaj, 1, ZRowMaj);
}

template <typename ElementType>
void UnrolledOutputConvolutionRowMaj(const ElementType* WRowMaj, const ElementType* XChlMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    throw std::invalid_argument("Not yet implemented");
}