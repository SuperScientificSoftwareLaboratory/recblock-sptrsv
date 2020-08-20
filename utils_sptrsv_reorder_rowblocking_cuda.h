#ifndef _UTILS_SPTRSV_REORDER_ROWBLOCKING_CUDA_
#define _UTILS_SPTRSV_REORDER_ROWBLOCKING_CUDA_

#include "common.h"
#include "utils.h"
#include "utils_reordering.h"
#include "utils_sptrsv_cuda.h"
#include "utils_spmv_cuda.h"
#include "findlevel.h"

int sptrsv_reorder_rowblocking_cuda(const int           *cscColPtrTR,
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

    // transpose to CSR format
    int *csrRowPtrTR = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxTR = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *csrValTR = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);
    matrix_transposition(n, m, nnzTR,
                        cscColPtrTR, cscRowIdxTR, cscValTR,
                        csrColIdxTR, csrRowPtrTR, csrValTR);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    const int m_slice = m / (nlevel+1);
    VALUE_TYPE *b_temp = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    memcpy(b_temp, b, n * sizeof(VALUE_TYPE));
    
    for (int li = 0; li < nlevel+1; li++)
    {
        // step 0. divide the CSR triangular slice into a CSR rectangular and a CSC triangulars
        // left rectangular
        int m_rec = li == nlevel ? (m - nlevel * m_slice) : m_slice;
        int n_rec = li * m_slice; 
        int nnz_rec = 0;

        // right triangular
        int m_tri = m_rec;
        int n_tri = m_tri;
        int nnz_tri = 0;

        int *csrRowPtr_rec = (int *)malloc((m_rec+1) * sizeof(int));
        int *csrRowPtr_tri = (int *)malloc((m_tri+1) * sizeof(int));

        // check nnz of the left part (top tri and rec)
        gettimeofday(&t1, NULL);
        csrRowPtr_rec[0] = 0;
        csrRowPtr_tri[0] = 0;
        for (int i = 0; i < m_rec; i++)
        {
            int nnzr_tri = 0;
            int nnzr_rec = 0;
            for (int j = csrRowPtrTR[li * m_slice + i]; j < csrRowPtrTR[li * m_slice + i+1]; j++)
            {
                int colidx = csrColIdxTR[j];
                if (colidx < li * m_slice)
                    nnzr_rec++;
                else
                    nnzr_tri++;
            }
            csrRowPtr_rec[i+1] = csrRowPtr_rec[i] + nnzr_rec;
            csrRowPtr_tri[i+1] = csrRowPtr_tri[i] + nnzr_tri;
        }
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        nnz_rec = csrRowPtr_rec[m_rec];
        nnz_tri = csrRowPtr_tri[m_tri];

        int *csrColIdx_rec = (int *)malloc(nnz_rec * sizeof(int));
        VALUE_TYPE *csrVal_rec = (VALUE_TYPE *)malloc(nnz_rec * sizeof(VALUE_TYPE));

        int *csrColIdx_tri = (int *)malloc(nnz_tri * sizeof(int));
        VALUE_TYPE *csrVal_tri = (VALUE_TYPE *)malloc(nnz_tri * sizeof(VALUE_TYPE));

        // copy nonzeros into sub-matrix (rec and tri)
        gettimeofday(&t1, NULL);
        for (int i = 0; i < m_rec; i++)
        {
            int nnzr_rec = 0;
            int nnzr_tri = 0;
            int off_rec = csrRowPtr_rec[i];
            int off_tri = csrRowPtr_tri[i];
            
            for (int j = csrRowPtrTR[li * m_slice + i]; j < csrRowPtrTR[li * m_slice + i+1]; j++)
            {
                int colidx = csrColIdxTR[j];
                VALUE_TYPE val = csrValTR[j];
                if (colidx < li * m_slice)
                {
                    csrColIdx_rec[off_rec + nnzr_rec] = colidx;
                    csrVal_rec[off_rec + nnzr_rec] = val;
                    nnzr_rec++;
                }
                else
                {
                    csrColIdx_tri[off_tri + nnzr_tri] = colidx - n_rec;
                    csrVal_tri[off_tri + nnzr_tri] = val;
                    nnzr_tri++;
                }
            }
        }
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 1. compute SpMV of rec
        // transpose rec to CSR format for faster SpMV 
        VALUE_TYPE *x_rec = &x[0];
        VALUE_TYPE *y_rec = (VALUE_TYPE *)malloc(m_rec * sizeof(VALUE_TYPE));
        memset(y_rec, 0, m_rec * sizeof(VALUE_TYPE));
        spmv_warpvec_csr_cuda(csrRowPtr_rec, csrColIdx_rec, csrVal_rec, 
                                m_rec, n_rec, nnz_rec,
                                rhs, opt, x_rec, y_rec, nnzsum_spmv, timesum_spmv, preprocess);

        for (int i = 0; i < m_rec; i++)
            b_temp[li * m_slice + i] -= y_rec[i];

        free(csrRowPtr_rec);
        free(csrColIdx_rec);
        free(csrVal_rec);
        free(y_rec);
        //printf("Rec is done\n");

        // step 2. compute SpTRSV of tri

        // transpose rec to CSC format for SpTRSV
        int *cscColPtr_tri = (int *)malloc((n_tri+1) * sizeof(int));
        int *cscRowIdx_tri = (int *)malloc(nnz_tri * sizeof(int));
        VALUE_TYPE *cscVal_tri = (VALUE_TYPE *)malloc(nnz_tri * sizeof(VALUE_TYPE));

        gettimeofday(&t1, NULL);
        matrix_transposition(m_tri, n_tri, nnz_tri,
                             csrRowPtr_tri, csrColIdx_tri, csrVal_tri,
                             cscRowIdx_tri, cscColPtr_tri, cscVal_tri);
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // reorder top tri
        int *levelItem_tri = (int *)malloc(n_tri * sizeof(int));
        int *cscColPtr_tri_new = (int *)malloc((n_tri+1) * sizeof(int));
        int *cscRowIdx_tri_new = (int *)malloc(nnz_tri * sizeof(int));
        VALUE_TYPE *cscVal_tri_new = (VALUE_TYPE *)malloc(nnz_tri * sizeof(VALUE_TYPE));
        VALUE_TYPE *x_ref_tri_perm = (VALUE_TYPE *)malloc(n_tri * sizeof(VALUE_TYPE));
        VALUE_TYPE *b_tri_perm = (VALUE_TYPE *)malloc(m_tri * sizeof(VALUE_TYPE));
        VALUE_TYPE *x_tri_perm = (VALUE_TYPE *)malloc(n_tri * sizeof(VALUE_TYPE));
        memset(x_tri_perm, 0, n_tri * sizeof(VALUE_TYPE));

        gettimeofday(&t1, NULL);
        levelset_reordering_colrow_csc(cscColPtr_tri, cscRowIdx_tri, cscVal_tri, 
                            cscColPtr_tri_new, cscRowIdx_tri_new, cscVal_tri_new, 
                            levelItem_tri, m_tri, n_tri, nnz_tri, substitution);
        gettimeofday(&t2, NULL);
        *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        const VALUE_TYPE *b_tri = &b_temp[li * m_slice];
        const VALUE_TYPE *x_ref_tri = &x_ref[li * m_slice];
        levelset_reordering_vecb(b_tri, x_ref_tri, b_tri_perm, x_ref_tri_perm, levelItem_tri, m_tri);

        //printf("nlevel = %i\n", nlevel);

        sptrsv_syncfree_csc_cuda(cscColPtr_tri_new, cscRowIdx_tri_new, cscVal_tri_new, 
                            m_tri, n_tri, nnz_tri, substitution, rhs, OPT_WARP_AUTO, 
                            x_tri_perm, b_tri_perm, x_ref_tri_perm, nnzsum_sptrsv, timesum_sptrsv, preprocess);

        // permute x 
        VALUE_TYPE *x_tri = &x[li * m_slice];
        levelset_reordering_vecx(x_tri_perm, x_tri, levelItem_tri, n_tri);

        free(cscColPtr_tri);
        free(cscRowIdx_tri);
        free(cscVal_tri);
        free(levelItem_tri);
        free(cscColPtr_tri_new);
        free(cscRowIdx_tri_new);
        free(cscVal_tri_new);
        free(x_ref_tri_perm);
        free(b_tri_perm);
        free(x_tri_perm);
        free(csrRowPtr_tri);
        free(csrColIdx_tri);
        free(csrVal_tri);
        //printf("Top Tri is done\n");

    }

    free(b_temp);
    free(csrRowPtrTR);
    free(csrColIdxTR);
    free(csrValTR);

    return 0;
}
#endif
