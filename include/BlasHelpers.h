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
void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc);
void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
