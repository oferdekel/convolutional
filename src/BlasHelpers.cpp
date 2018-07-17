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

void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
{
    CBLAS_ORDER blasOrder = (order == MatrixOrder::rowMajor) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    CBLAS_TRANSPOSE blasTransposeA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    CBLAS_TRANSPOSE blasTransposeB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    cblas_sgemm(blasOrder, blasTransposeA, blasTransposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

MatrixOrder flipOrder(MatrixOrder order)
{
    if (order == MatrixOrder::RowMajor)
    {
        return MatrixOrder::ColumnMajor;
    }
    return MatrixOrder::RowMajor;
}

// void Gemm(MatrixOrder order, bool transposeA, bool transposeB, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
// {
//     auto orderA = transposeA ? flipOrder(order) : order;
//     auto orderB = transposeB ? flipOrder(order) : order;
//     auto orderC = order;

//     auto AMat = MatrixConstInterface<float>(A, { m, k }, {0,0}); // TODO
//     auto BMat = MatrixConstInterface<float>(B, { k, n }, {0,0});
//     auto CMat = MatrixInterface<float>(C, { m, n }, {0,0});

//     for (size_t i = 0; i < CMat.Size(0); ++i)
//     {
//         for (size_t j = 0; j < CMat.Size(1); ++j)
//         {
//             float value = 0;
//             for (size_t k = 0; k < AMat.Size(1); ++k)
//             {
//                 value += AMat({i, k}) * BMat({k, j});
//             }
//             CMat({i, j}) = value;
//         }
//     }
// }

#endif

// void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float beta, float* C, size_t ldc)
// {
//     auto transposeA = (orderA == orderC) ? false : true;
//     auto transposeB = (orderB == orderC) ? false : true;
//     Gemm(orderC, transposeA, transposeB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
// }

// void Gemm(MatrixOrder orderA, MatrixOrder orderB, MatrixOrder orderC, size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C)
// {
//     size_t lda = (orderA == MatrixOrder::RowMajor) ? k : m;
//     size_t ldb = (orderB == MatrixOrder::RowMajor) ? n : k;
//     size_t ldc = (orderC == MatrixOrder::RowMajor) ? n : m;
//     Gemm(orderA, orderB, orderC, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
// }

