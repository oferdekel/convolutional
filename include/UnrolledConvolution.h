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
                int urow = yRow * yCols + yCol;
                float* target = U + (urow * wRows + wRow) * wCols * wChls;
                
                // calculate memcpy source
                int xRow = yRow * vStride + wRow;
                int xCol = yCol * hStride;
                const float* source = XRowMaj + (xRow * xCols + xCol) * xChls;
                
                // copy from X to U
                memcpy(target, source, copySize);
            }  
        }   
    }   
}

template <typename ElementType>
void UnrolledConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    std::vector<ElementType> U(100);
    UnrollInput(XRowMaj, U.data(), wRows, wCols, wChls, vStride, hStride, yRows, yCols);

    Tensor<ElementType,3> UMat({3,3,3}, RowMaj3Order);
    std::cout << *UMat.Data();
    //std::cout << UMat({0,0});
    // std::cout << UMat({1,0});
    // std::cout << UMat({2,0});
    // std::cout << UMat({0,1});
    // std::cout << UMat({1,1});
    // std::cout << UMat({2,1});
    // std::cout << UMat({0,2});
    // std::cout << UMat({1,2});
    // std::cout << UMat({2,2});

    //std::cout << UMat << std::endl;
}
