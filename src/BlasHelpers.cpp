////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"

#ifdef BLAS

// BLAS
#include <cblas.h>

void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    CBLAS_ORDER blasOrder = (order == MatrixOrder::rowMajor) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    cblas_sgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, double alpha, const double* A, size_t lda, const double* B, size_t ldb, double beta, double* C, size_t ldc)
{
    CBLAS_ORDER blasOrder = (order == MatrixOrder::rowMajor) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    cblas_dgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    auto orderA = transposeA ? flipOrder(order) : order;
    auto orderB = transposeB ? flipOrder(order) : order;
    auto orderC = order;

    auto AMat = MatrixConstInterface<float>(A, m, k, orderA);
    auto BMat = MatrixConstInterface<float>(B, k, n, orderB);
    auto CMat = MatrixInterface<float>(C, m, n, orderC);

    for (size_t i = 0; i < CMat.NumRows(); ++i)
    {
        for (size_t j = 0; j < CMat.NumColumns(); ++j)
        {
            float value = 0;
            for (size_t k = 0; k < AMat.NumColumns(); ++k)
            {
                value += AMat(i, k) * BMat(k, j);
            }
            CMat(i, j) = value;
        }
    }
}

void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, double alpha, const double* A, size_t lda, const double* B, size_t ldb, double beta, double* C, size_t ldc)
{
}

#endif


void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    auto transposeA = (orderA == orderC) ? false : true;
    auto transposeB = (orderB == orderC) ? false : true;
    Gemm(orderC, transposeA, transposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, double alpha, const double* A, size_t lda, const double* B, size_t ldb, double beta, double* C, size_t ldc)
{
    auto transposeA = (orderA == orderC) ? false : true;
    auto transposeB = (orderB == orderC) ? false : true;
    Gemm(orderC, transposeA, transposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C)
{
    size_t lda = (orderA == MatrixOrder::RowMajor) ? k : m;
    size_t ldb = (orderB == MatrixOrder::RowMajor) ? n : k;
    size_t ldc = (orderC == MatrixOrder::RowMajor) ? n : m;
    Gemm(orderA, orderB, orderC, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, double alpha, const double* A, const double* B, double beta, double* C)
{
    size_t lda = (orderA == MatrixOrder::RowMajor) ? k : m;
    size_t ldb = (orderB == MatrixOrder::RowMajor) ? n : k;
    size_t ldc = (orderC == MatrixOrder::RowMajor) ? n : m;
    Gemm(orderA, orderB, orderC, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

