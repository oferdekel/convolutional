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

// GEMM overloads
void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C);
