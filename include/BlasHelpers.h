////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cassert>

#include "Matrix.h"

// GEMM
void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, double alpha, const double* A, size_t lda, const double* B, size_t ldb, double beta, double* C, size_t ldc);

void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, double alpha, const double* A, size_t lda, const double* B, size_t ldb, double beta, double* C, size_t ldc);

void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, double alpha, const double* A, const double* B, double beta, double* C);

template<typename ElementType>
void Gemm(const MatrixConstInterface<ElementType>& A, const MatrixConstInterface<ElementType>& B, MatrixInterface<ElementType>& C);

//
//
//

template<typename ElementType>
void Gemm(const MatrixConstInterface<ElementType>& A, const MatrixConstInterface<ElementType>& B, MatrixInterface<ElementType>& C)
{
    assert(A.NumColumns() == B.NumRows() && A.NumRows() == C.NumRows() && B.NumColumns() == C.NumColumns());
    Gemm(A.Order(), B.Order(), C.Order(), A.NumRows(), B.NumColumns(), A.NumColumns(), 1.0, A.Data(), B.Data(), 0.0, C.Data());
}