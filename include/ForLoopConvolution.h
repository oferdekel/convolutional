
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ForLoopConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

template <typename ElementType>
void ForLoopConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, ElementType* YRowMaj, int wCount, int wRows, int wCols, int wChls, int vStride, int hStride, int yRows, int yCols)
{
    int yChls = wCount;
    int xRows = (yRows - 1) * vStride + wRows;
    int xCols = (yCols - 1) * hStride + wCols;
    int xChls = wChls;

    for (int yRow = 0; yRow < yRows; ++yRow)
    {
        for (int yCol = 0; yCol < yCols; ++yCol)
        {
            for (int yChl = 0; yChl < yChls; ++yChl)
            {
                ElementType output = 0;
                for (int wRow = 0; wRow < wRows; ++wRow)
                {
                    for (int wCol = 0; wCol < wCols; ++wCol)
                    {
                        for (int wChl = 0; wChl < wChls; ++wChl)
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
