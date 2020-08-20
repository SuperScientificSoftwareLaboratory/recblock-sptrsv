#ifndef _SPTRSV_REBLOCKING_LV_CUDA_
#define _SPTRSV_REBLOCKING_LV_CUDA_

#include "common.h"
#include "utils_sptrsv_recblocking_cuda.h"
#include <cuda_runtime.h>

int sptrsv_reblocking_lv_cuda(const int           *cscColPtrTR,
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
    // call normal reblocking SpTRSV
    sptrsv_recblocking_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                         substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, nlevel, 
                         nnzsum_sptrsv, nnzsum_spmv, timesum_sptrsv, timesum_spmv, preprocess);

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
        printf("sptrsv_reblocking_lv_cuda: x check passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrsv_reblocking_lv_cuda: x check _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    return 0;
}

#endif



