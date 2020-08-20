#ifndef _UTILS_SPTRSV_REORDER_COLBLOCKING_CUDA_
#define _UTILS_SPTRSV_REORDER_COLBLOCKING_CUDA_

#include "common.h"
#include "utils.h"
#include "utils_reordering.h"
#include "utils_sptrsv_cuda.h"
#include "utils_spmv_cuda.h"
#include "findlevel.h"

int sptrsv_reorder_colblocking_cuda(const int           *cscColPtrTR,
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

    const int m_slice = m / (nlevel+1);
    VALUE_TYPE *b_temp = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    memcpy(b_temp, b, n * sizeof(VALUE_TYPE));
    
    for (int li = 0; li < nlevel+1; li++)
    {
        // step 0. divide the CSC triangular slice into a CSC triangulars ans a CSR rectangular
        // top triangular
        int m_toptri = li == nlevel ? (m - nlevel * m_slice) : m_slice;
        int n_toptri = m_toptri;
        int nnz_toptri = 0;

        // rectangular
        int m_rec = m - (li * m_slice + m_toptri); 
        int n_rec = n_toptri; 
        int nnz_rec = 0;

        int *cscColPtr_toptri = (int *)malloc((n_toptri+1) * sizeof(int));
        int *cscColPtr_rec = (int *)malloc((n_rec+1) * sizeof(int));

        // check nnz of the left part (top tri and rec)
        gettimeofday(&t1, NULL);
        cscColPtr_toptri[0] = 0;
        cscColPtr_rec[0] = 0;
        for (int i = 0; i < n_rec; i++)
        {
            int nnzr_toptri = 0;
            int nnzr_rec = 0;
            for (int j = cscColPtrTR[li * m_slice + i]; j < cscColPtrTR[li * m_slice + i+1]; j++)
            {
                int rowidx = cscRowIdxTR[j] - li * m_slice;
                if (rowidx < m_toptri)
                    nnzr_toptri++;
                else
                    nnzr_rec++;
            }
            cscColPtr_toptri[i+1] = cscColPtr_toptri[i] + nnzr_toptri;
            cscColPtr_rec[i+1] = cscColPtr_rec[i] + nnzr_rec;
        }
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        nnz_toptri = cscColPtr_toptri[n_toptri];
        nnz_rec = cscColPtr_rec[n_rec];

        int *cscRowIdx_toptri = (int *)malloc(nnz_toptri * sizeof(int));
        VALUE_TYPE *cscVal_toptri = (VALUE_TYPE *)malloc(nnz_toptri * sizeof(VALUE_TYPE));

        int *cscRowIdx_rec = (int *)malloc(nnz_rec * sizeof(int));
        VALUE_TYPE *cscVal_rec = (VALUE_TYPE *)malloc(nnz_rec * sizeof(VALUE_TYPE));

        // copy nonzeros into sub-matrix (top tri and rec)
        gettimeofday(&t1, NULL);
        for (int i = 0; i < n_rec; i++)
        {
            int nnzr_toptri = 0;
            int nnzr_rec = 0;
            int off_toptri = cscColPtr_toptri[i];
            int off_rec = cscColPtr_rec[i];
            for (int j = cscColPtrTR[li * m_slice + i]; j < cscColPtrTR[li * m_slice + i+1]; j++)
            {
                int rowidx = cscRowIdxTR[j] - li * m_slice;
                VALUE_TYPE val = cscValTR[j];
                if (rowidx < m_toptri)
                {
                    cscRowIdx_toptri[off_toptri + nnzr_toptri] = rowidx;
                    cscVal_toptri[off_toptri + nnzr_toptri] = val;
                    nnzr_toptri++;
                }
                else
                {
                    cscRowIdx_rec[off_rec + nnzr_rec] = rowidx - n_rec;
                    cscVal_rec[off_rec + nnzr_rec] = val;
                    nnzr_rec++;
                }
            }
        }
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

        const VALUE_TYPE *b_toptri = &b_temp[li * m_slice];
        const VALUE_TYPE *x_ref_toptri = &x_ref[li * m_slice];
        gettimeofday(&t1, NULL);
        levelset_reordering_vecb(b_toptri, x_ref_toptri, b_toptri_perm, x_ref_toptri_perm, levelItem_toptri, m_toptri);
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        //printf("nlevel = %i\n", nlevel);

        sptrsv_syncfree_csc_cuda(cscColPtr_toptri_new, cscRowIdx_toptri_new, cscVal_toptri_new, 
                            m_toptri, n_toptri, nnz_toptri, substitution, rhs, OPT_WARP_AUTO, 
                            x_toptri_perm, b_toptri_perm, x_ref_toptri_perm, nnzsum_sptrsv, timesum_sptrsv, preprocess);

        // permute x 
        VALUE_TYPE *x_toptri = &x[li * m_slice];
        gettimeofday(&t1, NULL);
        levelset_reordering_vecx(x_toptri_perm, x_toptri, levelItem_toptri, n_toptri);
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        free(cscColPtr_toptri);
        free(cscRowIdx_toptri);
        free(cscVal_toptri);
        free(levelItem_toptri);
        free(cscColPtr_toptri_new);
        free(cscRowIdx_toptri_new);
        free(cscVal_toptri_new);
        free(x_ref_toptri_perm);
        free(b_toptri_perm);
        free(x_toptri_perm);
        //printf("Top Tri is done\n");

        // step 2. compute SpMV of rec
        // transpose rec to CSR format for faster SpMV 
        int *csrRowPtr_rec = (int *)malloc((m_rec+1) * sizeof(int));
        int *csrColIdx_rec = (int *)malloc(nnz_rec * sizeof(int));
        VALUE_TYPE *csrVal_rec = (VALUE_TYPE *)malloc(nnz_rec * sizeof(VALUE_TYPE));

        gettimeofday(&t1, NULL);
        matrix_transposition(n_rec, m_rec, nnz_rec,
                            cscColPtr_rec, cscRowIdx_rec, cscVal_rec,
                            csrColIdx_rec, csrRowPtr_rec, csrVal_rec);
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        VALUE_TYPE *x_rec = &x[li * m_slice];
        VALUE_TYPE *y_rec = (VALUE_TYPE *)malloc(m_rec * sizeof(VALUE_TYPE));
        memset(y_rec, 0, m_rec * sizeof(VALUE_TYPE));
        spmv_warpvec_csr_cuda(csrRowPtr_rec, csrColIdx_rec, csrVal_rec, 
                                m_rec, n_rec, nnz_rec,
                                rhs, opt, x_rec, y_rec, nnzsum_spmv, timesum_spmv, preprocess);

        for (int i = 0; i < m_rec; i++)
            b_temp[li * m_slice + m_toptri + i] -= y_rec[i];

        free(csrRowPtr_rec);
        free(csrColIdx_rec);
        free(csrVal_rec);
        free(cscColPtr_rec);
        free(cscRowIdx_rec);
        free(cscVal_rec);
        free(y_rec);
        //printf("Rec is done\n");
    }
    free(b_temp);

    return 0;
}
#endif
