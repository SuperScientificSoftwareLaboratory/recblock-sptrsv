#ifndef _SPTRSV_REORDER_ROWBLOCKING_LV_CUDA_
#define _SPTRSV_REORDER_ROWBLOCKING_LV_CUDA_

#include "common.h"
#include "findlevel.h"
#include "utils_sptrsv_reorder_rowblocking_cuda.h"
#include <cuda_runtime.h>

int sptrsv_reorder_rowblocking_lv_cuda(const int           *cscColPtrTR,
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
    struct timeval t1, t2;

    int *levelItem = (int *)malloc(m * sizeof(int));
    int *cscColPtrTR_new = (int *)malloc((n+1) * sizeof(int));
    int *cscRowIdxTR_new = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *cscValTR_new = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

    VALUE_TYPE *x_perm = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    memset(x_perm, 0, n * sizeof(VALUE_TYPE));

    VALUE_TYPE *x_ref_perm = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    //memset(x_ref_perm, 0, n * sizeof(VALUE_TYPE));

    VALUE_TYPE *b_perm = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    //memset(b_perm, 0, m * sizeof(VALUE_TYPE));

    // reorder input CSC according to level-set order
    gettimeofday(&t1, NULL);
    levelset_reordering_colrow_csc(cscColPtrTR, cscRowIdxTR, cscValTR, 
                        cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new, 
                        levelItem, m, n, nnzTR, substitution);

    levelset_reordering_vecb(b, x_ref, b_perm, x_ref_perm, levelItem, m);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // call normal row blocking SpTRSV
    sptrsv_reorder_rowblocking_cuda(cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new, m, n, nnzTR,
                         substitution, rhs, OPT_WARP_AUTO, x_perm, b_perm, x_ref_perm, 
                         nlevel, nnzsum_sptrsv, nnzsum_spmv, timesum_sptrsv, timesum_spmv, preprocess);

    free(cscColPtrTR_new);
    free(cscRowIdxTR_new);
    free(cscValTR_new);
    free(x_ref_perm);
    free(b_perm);

    // permute x 
    gettimeofday(&t1, NULL);
    levelset_reordering_vecx(x_perm, x, levelItem, n);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    free(levelItem);
    free(x_perm);

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrsv_reorder_rowblocking_lv_cuda: x check passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrsv_reorder_rowblocking_lv_cuda: x check _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    return 0;
}

#endif



