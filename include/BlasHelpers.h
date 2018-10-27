////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Tensor.h"

// GEMM overloads
void Gemm(MatrixOrder matrixOrderC, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
void Gemm(MatrixOrder matrixOrderA, MatrixOrder matrixOrderB, MatrixOrder matrixOrderC, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
void Gemm(MatrixOrder matrixOrderA, MatrixOrder matrixOrderB, MatrixOrder matrixOrderC, int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C);

// AXPY 
void Axpy(int n, float alpha, const float* X, int incX, float* Y, int incY);

// COPY
void Copy(int n, const float* X, int incX, float* Y, int incY);
