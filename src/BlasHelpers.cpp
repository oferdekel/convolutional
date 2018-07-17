////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"
#include "Tensor.h"

#ifdef USE_BLAS

// BLAS
#include <cblas.h>

void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    CBLAS_ORDER blasOrder = isRowMajor ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    cblas_sgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    MatrixOrder orderA = isRowMajor ^ transposeA ? RowMajorMatrixOrder : ColumnMajorMatrixOrder;
    MatrixOrder orderB = isRowMajor ^ transposeB ? RowMajorMatrixOrder : ColumnMajorMatrixOrder;
    MatrixOrder orderC = isRowMajor ? RowMajorMatrixOrder : ColumnMajorMatrixOrder;

    auto AMat = MatrixConstInterface<float>(A, { m, k }, orderA); 
    auto BMat = MatrixConstInterface<float>(B, { k, n }, orderB);
    auto CMat = MatrixInterface<float>(C, { m, n }, orderC);

    for (size_t i = 0; i < CMat.Size(0); ++i)
    {
        for (size_t j = 0; j < CMat.Size(1); ++j)
        {
            float value = 0;
            for (size_t k = 0; k < AMat.Size(1); ++k)
            {
                value += AMat({i, k}) * BMat({k, j});
            }
            CMat({i, j}) = value;
        }
    }
}

#endif

void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    BLASGemm(isCRowMajor, isARowMajor ^ isCRowMajor, isBRowMajor ^ isCRowMajor, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C)
{
    size_t lda = isARowMajor ? k : m;
    size_t ldb = isBRowMajor ? n : k;
    size_t ldc = isCRowMajor ? n : m;
    Gemm(isARowMajor, isBRowMajor, isCRowMajor, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

