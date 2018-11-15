////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     BlasHelpers.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BlasHelpers.h"

// stl
#include <iostream>

#ifdef USE_BLAS
#include BLAS_HEADER_FILE

void Gemm(MatrixOrder matrixOrderC, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    if(matrixOrderC == RowMaj && transposeA && !transposeB)
    {
        // workaround for bug in OpenBLAS 3.4
        // equivalent formulation for this line:
        // cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);       
        cblas_sgemm(CBLAS_ORDER::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
    }
    else
    {
        CBLAS_ORDER blasOrder = (matrixOrderC == RowMaj) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
        CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
        CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

        cblas_sgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

void Axpy(int n, float alpha, const float* X, int incX, float* Y, int incY)
{
    cblas_saxpy(n, alpha, X, incX, Y, incY);
}

void Copy(int n, const float* X, int incX, float* Y, int incY)
{
    cblas_scopy(n, X, incX, Y, incY);
}

#else

void Gemm(MatrixOrder matrixOrderC, bool transposeA, bool transposeB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    MatrixOrder matrixOrderA = (matrixOrderC == RowMaj) ^ transposeA ? RowMaj : ColMaj;
    MatrixOrder matrixOrderB = (matrixOrderC == RowMaj) ^ transposeB ? RowMaj : ColMaj;

    auto AMat = MatrixConstInterface<float>(A, { m, k }, matrixOrderA); 
    auto BMat = MatrixConstInterface<float>(B, { k, n }, matrixOrderB);
    auto CMat = MatrixInterface<float>(C, { m, n }, matrixOrderC);
    
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
}

void Axpy(int n, float alpha, const float* X, int incX, float* Y, int incY)
{
    for(int i=0; i < n; ++i)
    {
        Y[i * incY] += alpha * X[i * incX];
    }
}

void Copy(int n, const float* X, int incX, float* Y, int incY)
{
    for(int i=0; i < n; ++i)
    {
        Y[i * incY] = X[i * incX];
    }    
}

void PrintBlasInfo()
{
    std::cout << "BLAS not used\n";
}

#endif

void Gemm(MatrixOrder matrixOrderA, MatrixOrder matrixOrderB, MatrixOrder matrixOrderC, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    Gemm(matrixOrderC, (matrixOrderA != matrixOrderC), (matrixOrderB != matrixOrderC), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void Gemm(MatrixOrder matrixOrderA, MatrixOrder matrixOrderB, MatrixOrder matrixOrderC, int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C)
{
    int lda = (matrixOrderA == RowMaj) ? k : m;
    int ldb = (matrixOrderB == RowMaj) ? n : k;
    int ldc = (matrixOrderC == RowMaj) ? n : m;
    Gemm(matrixOrderA, matrixOrderB, matrixOrderC, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

