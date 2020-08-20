#ifndef _UTILS_SPTRSV_REORDER_REBLOCKING_CUDA_
#define _UTILS_SPTRSV_REORDER_REBLOCKING_CUDA_

#include "common.h"
#include "utils.h"
#include "utils_reordering.h"
#include "utils_sptrsv_cuda.h"
#include "utils_spmv_cuda.h"
#include "findlevel.h"

int sptrsv_reorder_recblocking_cuda(const int           *cscColPtrTR,
                              const int           *cscRowIdxTR,
                              const VALUE_TYPE    *cscValTR,
                              const int            m,
                              const int            n,
                              const int            nnzTR,
                              const int            substitution,
                              const int            rhs,
                              const int            opt,
                                    VALUE_TYPE    *x,
                              const VALUE_TYPE    *b,
                              const VALUE_TYPE    *x_ref,
                              const int            nlevel,
                                    int           *nnzsum_sptrsv, 
                                    int           *nnzsum_spmv,
                                    double        *timesum_sptrsv,
                                    double        *timesum_spmv,
                                    double        *preprocess)
{
    if (nlevel == 0)
    {
        printf("#level = 0. Please run the normal SpTRSV. Program return.\n");
        return 0;
    }
    struct timeval t1, t2;

    // step 0. divide the CSC triangular into two CSC triangulars ans one CSR square
    // top triangular
    int m_toptri = m / 2; 
    int n_toptri = n / 2; 
    int nnz_toptri = 0;

    // square
    int m_sqr = m - m_toptri; 
    int n_sqr = n_toptri;
    int nnz_sqr = 0;

    // bottom triangular
    int m_bottri = m - m_toptri; 
    int n_bottri = n - n_toptri;
    int nnz_bottri = 0;

    int *cscColPtr_toptri = (int *)malloc((n_toptri+1) * sizeof(int));
    int *cscColPtr_sqr = (int *)malloc((n_sqr+1) * sizeof(int));
    int *cscColPtr_bottri = (int *)malloc((n_bottri+1) * sizeof(int));

    // check nnz of the left part (top tri and sqr)
    gettimeofday(&t1, NULL);
    cscColPtr_toptri[0] = 0;
    cscColPtr_sqr[0] = 0;
    for (int i = 0; i < n_sqr; i++)
    {
        int nnzr_toptri = 0;
        int nnzr_sqr = 0;
        for (int j = cscColPtrTR[i]; j < cscColPtrTR[i+1]; j++)
        {
            int rowidx = cscRowIdxTR[j];
            if (rowidx < m_toptri)
                nnzr_toptri++;
            else
                nnzr_sqr++;
        }
        cscColPtr_toptri[i+1] = cscColPtr_toptri[i] + nnzr_toptri;
        cscColPtr_sqr[i+1] = cscColPtr_sqr[i] + nnzr_sqr;
    }

    // check nnz of the right part (bottom tri)
    cscColPtr_bottri[0] = 0;
    for (int i = n_sqr; i < n; i++)
    {
        cscColPtr_bottri[i-n_sqr+1] = cscColPtr_bottri[i-n_sqr] + cscColPtrTR[i+1] - cscColPtrTR[i];
    }
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    nnz_toptri = cscColPtr_toptri[n_toptri];
    nnz_sqr = cscColPtr_sqr[n_sqr];
    nnz_bottri = cscColPtr_bottri[n_bottri];
    //printf("nnz_toptri = %i, nnz_sqr = %i, nnz_bottri = %i\n", nnz_toptri, nnz_sqr, nnz_bottri);

    int *cscRowIdx_toptri = (int *)malloc(nnz_toptri * sizeof(int));
    VALUE_TYPE *cscVal_toptri = (VALUE_TYPE *)malloc(nnz_toptri * sizeof(VALUE_TYPE));

    int *cscRowIdx_sqr = (int *)malloc(nnz_sqr * sizeof(int));
    VALUE_TYPE *cscVal_sqr = (VALUE_TYPE *)malloc(nnz_sqr * sizeof(VALUE_TYPE));
    
    int *cscRowIdx_bottri = (int *)malloc(nnz_bottri * sizeof(int));
    VALUE_TYPE *cscVal_bottri = (VALUE_TYPE *)malloc(nnz_bottri * sizeof(VALUE_TYPE));

    // copy nonzeros into sub-matrix (top tri and sqr)
    gettimeofday(&t1, NULL);
    for (int i = 0; i < n_sqr; i++)
    {
        int nnzr_toptri = 0;
        int nnzr_sqr = 0;
        int off_toptri = cscColPtr_toptri[i];
        int off_sqr = cscColPtr_sqr[i];
        for (int j = cscColPtrTR[i]; j < cscColPtrTR[i+1]; j++)
        {
            int rowidx = cscRowIdxTR[j];
            VALUE_TYPE val = cscValTR[j];
            if (rowidx < m_toptri)
            {
                cscRowIdx_toptri[off_toptri + nnzr_toptri] = rowidx;
                cscVal_toptri[off_toptri + nnzr_toptri] = val;
                nnzr_toptri++;
            }
            else
            {
                cscRowIdx_sqr[off_sqr + nnzr_sqr] = rowidx - n_sqr;
                cscVal_sqr[off_sqr + nnzr_sqr] = val;
                nnzr_sqr++;
            }
        }
    }
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // copy nonzeros into sub-matrix (bottom tri)
    gettimeofday(&t1, NULL);
    memcpy(cscRowIdx_bottri, &cscRowIdxTR[cscColPtrTR[n_sqr]], nnz_bottri * sizeof(int));
    for (int i = 0; i < nnz_bottri; i++)
        cscRowIdx_bottri[i] -= n_sqr;
    memcpy(cscVal_bottri, &cscValTR[cscColPtrTR[n_sqr]], nnz_bottri * sizeof(VALUE_TYPE));
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // step 1. compute SpTRSV of top tri
    // reorder top tri
    int *levelItem_toptri = (int *)malloc(n_toptri * sizeof(int));
    int *cscColPtr_toptri_new = (int *)malloc((n_toptri+1) * sizeof(int));
    int *cscRowIdx_toptri_new = (int *)malloc(nnz_toptri * sizeof(int));
    VALUE_TYPE *cscVal_toptri_new = (VALUE_TYPE *)malloc(nnz_toptri * sizeof(VALUE_TYPE));
    VALUE_TYPE *x_ref_toptri_perm = (VALUE_TYPE *)malloc(n_toptri * sizeof(VALUE_TYPE));
    VALUE_TYPE *b_toptri_perm = (VALUE_TYPE *)malloc(m_toptri * sizeof(VALUE_TYPE));
    VALUE_TYPE *x_toptri_perm = (VALUE_TYPE *)malloc(n_toptri * sizeof(VALUE_TYPE));
    memset(x_toptri_perm, 0, n_toptri * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);
    levelset_reordering_colrow_csc(cscColPtr_toptri, cscRowIdx_toptri, cscVal_toptri, 
                        cscColPtr_toptri_new, cscRowIdx_toptri_new, cscVal_toptri_new, 
                        levelItem_toptri, m_toptri, n_toptri, nnz_toptri, substitution);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    free(cscColPtr_toptri);
    free(cscRowIdx_toptri);
    free(cscVal_toptri);

    const VALUE_TYPE *b_toptri = &b[0];
    const VALUE_TYPE *x_ref_toptri = &x_ref[0];
    gettimeofday(&t1, NULL);
    levelset_reordering_vecb(b_toptri, x_ref_toptri, b_toptri_perm, x_ref_toptri_perm, levelItem_toptri, m_toptri);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    //printf("nlevel = %i\n", nlevel);

    if (nlevel == 1)
    {
        sptrsv_syncfree_csc_cuda(cscColPtr_toptri_new, cscRowIdx_toptri_new, cscVal_toptri_new, 
                               m_toptri, n_toptri, nnz_toptri, substitution, rhs, OPT_WARP_AUTO, 
                               x_toptri_perm, b_toptri_perm, x_ref_toptri_perm, nnzsum_sptrsv, timesum_sptrsv, preprocess);
    }
    else
    {
        sptrsv_reorder_recblocking_cuda(cscColPtr_toptri_new, cscRowIdx_toptri_new, cscVal_toptri_new, 
                               m_toptri, n_toptri, nnz_toptri, substitution, rhs, OPT_WARP_AUTO, 
                               x_toptri_perm, b_toptri_perm, x_ref_toptri_perm, nlevel-1, 
                               nnzsum_sptrsv, nnzsum_spmv, timesum_sptrsv, timesum_spmv, preprocess);
    }
    
    // permute x 
    VALUE_TYPE *x_toptri = &x[0];
    gettimeofday(&t1, NULL);
    levelset_reordering_vecx(x_toptri_perm, x_toptri, levelItem_toptri, n_toptri);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    free(levelItem_toptri);
    free(cscColPtr_toptri_new);
    free(cscRowIdx_toptri_new);
    free(cscVal_toptri_new);
    free(x_ref_toptri_perm);
    free(b_toptri_perm);
    free(x_toptri_perm);
    //printf("Top Tri is done\n");

    // step 2. compute SpMV of sqr
    // transpose sqr to CSR format for faster SpMV 
    int *csrRowPtr_sqr = (int *)malloc((m_sqr+1) * sizeof(int));
    int *csrColIdx_sqr = (int *)malloc(nnz_sqr * sizeof(int));
    VALUE_TYPE *csrVal_sqr = (VALUE_TYPE *)malloc(nnz_sqr * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);
    matrix_transposition(n_sqr, m_sqr, nnz_sqr,
                         cscColPtr_sqr, cscRowIdx_sqr, cscVal_sqr,
                         csrColIdx_sqr, csrRowPtr_sqr, csrVal_sqr);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    VALUE_TYPE *x_sqr = &x[0];
    VALUE_TYPE *y_sqr = (VALUE_TYPE *)malloc(m_sqr * sizeof(VALUE_TYPE));
    spmv_warpvec_csr_cuda(csrRowPtr_sqr, csrColIdx_sqr, csrVal_sqr, 
                             m_sqr, n_sqr, nnz_sqr,
                             rhs, opt, x_sqr, y_sqr, nnzsum_spmv, timesum_spmv, preprocess);
    free(csrRowPtr_sqr);
    free(csrColIdx_sqr);
    free(csrVal_sqr);
    free(cscColPtr_sqr);
    free(cscRowIdx_sqr);
    free(cscVal_sqr);
    //printf("Sqr is done\n");
    
    // step 3. compute SpTRSV of bottom tri

    // reorder top tri
    int *levelItem_bottri = (int *)malloc(n_bottri * sizeof(int));
    int *cscColPtr_bottri_new = (int *)malloc((n_bottri+1) * sizeof(int));
    int *cscRowIdx_bottri_new = (int *)malloc(nnz_bottri * sizeof(int));
    VALUE_TYPE *cscVal_bottri_new = (VALUE_TYPE *)malloc(nnz_bottri * sizeof(VALUE_TYPE));
    VALUE_TYPE *x_ref_bottri_perm = (VALUE_TYPE *)malloc(n_bottri * sizeof(VALUE_TYPE));
    VALUE_TYPE *b_temp_perm = (VALUE_TYPE *)malloc(m_bottri * sizeof(VALUE_TYPE));
    VALUE_TYPE *x_bottri_perm = (VALUE_TYPE *)malloc(n_bottri * sizeof(VALUE_TYPE));
    memset(x_bottri_perm, 0, n_bottri * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);
    levelset_reordering_colrow_csc(cscColPtr_bottri, cscRowIdx_bottri, cscVal_bottri, 
                        cscColPtr_bottri_new, cscRowIdx_bottri_new, cscVal_bottri_new, 
                        levelItem_bottri, m_bottri, n_bottri, nnz_bottri, substitution);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    free(cscColPtr_bottri);
    free(cscRowIdx_bottri);
    free(cscVal_bottri);

    const VALUE_TYPE *b_bottri = &b[m_toptri];
    VALUE_TYPE *b_temp = (VALUE_TYPE *)malloc(m_bottri * sizeof(VALUE_TYPE));
    for (int i = 0; i < m_bottri; i++)
        b_temp[i] = b_bottri[i] - y_sqr[i];
    const VALUE_TYPE *x_ref_bottri = &x_ref[m_toptri];

    gettimeofday(&t1, NULL);
    levelset_reordering_vecb(b_temp, x_ref_bottri, b_temp_perm, x_ref_bottri_perm, levelItem_bottri, m_bottri);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (nlevel == 1)
    {
        sptrsv_syncfree_csc_cuda(cscColPtr_bottri_new, cscRowIdx_bottri_new, cscVal_bottri_new, 
                               m_bottri, n_bottri, nnz_bottri, substitution, rhs, OPT_WARP_AUTO, 
                               x_bottri_perm, b_temp_perm, x_ref_bottri_perm, nnzsum_sptrsv, timesum_sptrsv, preprocess);
    }
    else
    {
        sptrsv_reorder_recblocking_cuda(cscColPtr_bottri_new, cscRowIdx_bottri_new, cscVal_bottri_new, 
                               m_bottri, n_bottri, nnz_bottri, substitution, rhs, OPT_WARP_AUTO, 
                               x_bottri_perm, b_temp_perm, x_ref_bottri_perm, nlevel-1, 
                               nnzsum_sptrsv, nnzsum_spmv, timesum_sptrsv, timesum_spmv, preprocess);
    }

    // permute x 
    VALUE_TYPE *x_bottri = &x[m_toptri];
    gettimeofday(&t1, NULL);
    levelset_reordering_vecx(x_bottri_perm, x_bottri, levelItem_bottri, n_bottri);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    
    free(cscColPtr_bottri_new);
    free(cscRowIdx_bottri_new);
    free(cscVal_bottri_new);
    free(y_sqr);
    free(b_temp);
    free(levelItem_bottri);
    free(x_ref_bottri_perm);
    free(b_temp_perm);
    free(x_bottri_perm);
    //printf("Bottom Tri is done\n");

    return 0;
}
#endif
