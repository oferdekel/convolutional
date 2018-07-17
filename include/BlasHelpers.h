////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cassert>
#include <cstddef>

#include "Matrix.h"


// Matrix Order
enum class MatrixOrder
{
    RowMajor = 0,
    ColumnMajor
};

// GEMM overloads
//void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
// void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
// void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
