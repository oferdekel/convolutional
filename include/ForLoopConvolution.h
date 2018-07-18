
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ForLoopConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

template <typename ElementType>
void ForLoopConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, int yRows, int yCols, int wRows, int wCols, int wChls, int vStride, int hStride)
{
    for (int yRow = 0; yRow < yRows; ++yRow)
    {
        for (int yCol = 0; yCol < yCols; ++yCol)
        {
            for (int yChl = 0; yChl < yChls; ++yChl)
            {
                ElementType value = 0;
                for (int wRow = 0; wRow < wRows; ++wRow)
                {
                    for (int wCol = 0; wCol < wCols; ++wCol)
                    {
                        for (int wChl = 0; wChl < wChls; ++wChl)
                        {
                            value += W({ yChl, wRow, wCol, wChl }) * X({ yRow * vStride + wRow, yCol * hStride + wCol, wChl });
                        }
                    }
                }
                Y({ yRow, yCol, yChl }) = value;
            }
        }
    }
}
