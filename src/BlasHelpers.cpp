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

void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    CBLAS_ORDER blasOrder = isRowMajor ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    cblas_sgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

void BLASGemm(bool isRowMajor, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    MatrixOrder orderA = isRowMajor ^ transposeA ? RowMaj2Order : ColMaj2Order;
    MatrixOrder orderB = isRowMajor ^ transposeB ? RowMaj2Order : ColMaj2Order;
    MatrixOrder orderC = isRowMajor ? RowMaj2Order : ColMaj2Order;

    auto AMat = MatrixConstInterface<float>(A, { m, k }, orderA); 
    auto BMat = MatrixConstInterface<float>(B, { k, n }, orderB);
    auto CMat = MatrixInterface<float>(C, { m, n }, orderC);

    //  std::cout << AMat << std::endl << std::endl;
    //  std::cout << BMat << std::endl << std::endl;
    
    for (int i = 0; i < CMat.Size(0); ++i)
    {
        for (int j = 0; j < CMat.Size(1); ++j)
        {
            float value = 0;
            for (int k = 0; k < AMat.Size(1); ++k)
            {
                value += AMat({i, k}) * BMat({k, j});
            }
            CMat({i, j}) = beta * CMat({i, j}) + alpha * value;
        }
    }
//    std::cout << CMat << std::endl << std::endl;
}

void BLASAxpy(int n, float alpha, const float* X, int incX, float* Y, int incY)
{
    for(int i=0; i < n; ++i)
    {
        Y[i * incY] += alpha * X[i * incX];
    }
}

#endif

void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    BLASGemm(isCRowMajor, isARowMajor ^ isCRowMajor, isBRowMajor ^ isCRowMajor, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(bool isARowMajor, bool isBRowMajor, bool isCRowMajor, int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C)
{
    int lda = isARowMajor ? k : m;
    int ldb = isBRowMajor ? n : k;
    int ldc = isCRowMajor ? n : m;
    Gemm(isARowMajor, isBRowMajor, isCRowMajor, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Axpy(int n, const float* X, float* Y)
{
    for(int i=0; i < n; ++i)
    {
        Y[i] += X[i];
    }
}

void Axpy(int n, float alpha, const float* X, int incX, float* Y, int incY)
{
    if(alpha == 1 && incX == 1 && incY == 1)
    {
        Axpy(n, X, Y);
    }
    else
    {
        BLASAxpy(n, alpha, X, incX, Y, incY);
    }
}

