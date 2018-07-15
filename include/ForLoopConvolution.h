
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     ForLoopConvolution.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

template <typename ElementType>
void ForLoopConvolution(const ElementType* WRowMaj, const ElementType* XRowMaj, int Yrows, int Ycols, int Wrows, int Wcols, int Wchls, int sv, int sh)
{

}




//     for (int r = 0; r < _a; ++r)
//     {
//         for (int s = 0; s < _b; ++s)
//         {
//             for (int t = 0; t < _c; ++t)
//             {
//                 ElementType value = 0;
//                 for (int i = 0; i < _l; ++i)
//                 {
//                     for (int j = 0; j < _m; ++j)
//                     {
//                         for (int k = 0; k < _n; ++k)
//                         {
//                             value += _W[t](i, j, k) * X(r * _p + i, s * _q + j, k);
//                         }
//                     }
//                 }


// void UnrollInput(float* U, const float* XRowMaj, int Yrows, int Ycols, int Wrows, int Wcols, int Wchls, int sv, int sh)
// {
