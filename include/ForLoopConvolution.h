
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ForLoopConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

template <typename ElementType>
void ForLoopConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, size_t wCount, size_t wRows, size_t wCols, size_t wChls, size_t vStride, size_t hStride, size_t yRows, size_t yCols)
{
    size_t yChls = wCount;
    size_t xRows = (yRows - 1) * vStride + wRows;
    size_t xCols = (yCols - 1) * hStride + wCols;
    size_t xChls = wChls;

    for (size_t yRow = 0; yRow < yRows; ++yRow)
    {
        for (size_t yCol = 0; yCol < yCols; ++yCol)
        {
            for (size_t yChl = 0; yChl < yChls; ++yChl)
            {
                ElementType output = 0;
                for (size_t wRow = 0; wRow < wRows; ++wRow)
                {
                    for (size_t wCol = 0; wCol < wCols; ++wCol)
                    {
                        for (size_t wChl = 0; wChl < wChls; ++wChl)
                        {
                            auto weight = *(WRowMaj + ((yChl * wRows + wRow) * wCols + wCol) * wChls + wChl);

                            auto xRow = yRow * vStride + wRow;
                            auto xCol = yCol * hStride + wCol;
                            auto xChl = wChl;
                            auto input = *(XRowMaj + (xRow * xCols + xCol) * xChls + xChl);

                            output += weight * input;
                        }
                    }
                }

                *(YRowMaj + (yRow * yCols + yCol) * yChls + yChl) = output;
            }
        }
    }
}
