#include "common.h"
#include <cuda_runtime.h>
#include "cusparse.h"

int sptrsv_cusparse(const int           *csrRowPtrL_tmp,
                  const int           *csrColIdxL_tmp,
                  const VALUE_TYPE    *csrValL_tmp,
                  const int            m,
                  const int            n,
                  const int            nnzL,
                        VALUE_TYPE    *x, 
                  const VALUE_TYPE    *b, 
                  const VALUE_TYPE    *x_ref,
                        double        *preprocessing_csrsv,
                        double        *runtime_csrsv,
                        double        *gflops_csrsv,
                        double        *preprocessing_csrsv2,
                        double        *runtime_csrsv2,
                        double        *gflops_csrsv2)
{
    // transfer host mem to device mem
    int *d_csrRowPtr;
    int *d_csrColInd;
    VALUE_TYPE *d_csrVal;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColInd, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_csrVal,    nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtr, csrRowPtrL_tmp, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColIdxL_tmp, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrValL_tmp,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n  * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));

    cusparseStatus_t status;
    cusparseHandle_t handle=0;

    status= cusparseCreate(&handle);

    // http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrsv2_solve
    // Suppose that L is m x m sparse matrix represented by CSR format, 
    // L is lower triangular with unit diagonal. 
    // Assumption:
    // - dimension of matrix L is m,
    // - matrix L has nnz number zero elements,
    // - handle is already created by cusparseCreate(),
    // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of L on device memory,
    // - d_b is right hand side vector on device memory,
    // - d_x is solution vector on device memory.

    cusparseMatDescr_t descr = 0;
    cusparseSolveAnalysisInfo_t csrsv_info = 0;
    csrsv2Info_t info = 0;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha_double = 1.;
    const float alpha_float = 1.;
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix L is base-0
    // - matrix L is lower triangular
    // - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
    //   (L may not have all diagonal elements.) 
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);





    //  - cuSPARSE SpTS (csrsv) analysis start!
    printf(" - cuSPARSE SpTS (csrsv) analysis start!\n");

    struct timeval t1, t2;

    // transpose from csr to csc first
    gettimeofday(&t1, NULL);

    // step 2: create a empty info structure
    cusparseCreateSolveAnalysisInfo(&csrsv_info);

    // step 3: perform analysis 
    for (int i = 0; i < 1; i++)
    {
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv_analysis(handle, trans, m, nnzL, descr, 
                                 (double*)d_csrVal, d_csrRowPtr, d_csrColInd,
                                 csrsv_info);   
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv_analysis(handle, trans, m, nnzL, descr, 
                                 (float*)d_csrVal, d_csrRowPtr, d_csrColInd,
                                 csrsv_info); 
    }
        
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_cusparse_csrsv_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cusparse_csrsv_analysis /= 1;
    *preprocessing_csrsv = time_cusparse_csrsv_analysis;

    printf("cuSPARSE SpTS (csrsv) analysis on L used %4.2f ms\n", time_cusparse_csrsv_analysis);

    //  - cuSPARSE SpTS (csrsv2) solve start!
    printf(" - cuSPARSE SpTS (csrsv) solve start!\n");

    // step 4: solve L*y = x
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv_solve(handle, trans, m, &alpha_double, descr,
                             (double *)d_csrVal, d_csrRowPtr, d_csrColInd, csrsv_info,
                             (double *)d_b, (double *)d_x);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv_solve(handle, trans, m, &alpha_float, descr,
                             (float *)d_csrVal, d_csrRowPtr, d_csrColInd, csrsv_info,
                             (float *)d_b, (float *)d_x);
    }
        
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_cusparse_csrsv_solve = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cusparse_csrsv_solve /= BENCH_REPEAT;
    *runtime_csrsv = time_cusparse_csrsv_solve;

    printf("cuSPARSE SpTS (csrsv) solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cusparse_csrsv_solve, 2*nnzL/(1e6*time_cusparse_csrsv_solve));
    *gflops_csrsv = 2*nnzL/(1e6*time_cusparse_csrsv_solve);

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    int err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
            err_counter++;
    }

    if (!err_counter)
        printf("cuSPARSE SpTS (csrsv) on L passed!\n");
    else
        printf("cuSPARSE SpTS (csrsv) on L failed!\n");








    printf("---------------------------------------------------------------------------------------------\n");





    //  - cuSPARSE SpTS (csrsv2) analysis start!
    printf(" - cuSPARSE SpTS (csrsv2) analysis start!\n");

    // transpose from csr to csc first
    gettimeofday(&t1, NULL);

    // step 2: create a empty info structure
    cusparseCreateCsrsv2Info(&info);

    // step 3: query how much memory used in csrsv2, and allocate the buffer
    for (int i = 0; i < 1; i++)
    {
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_bufferSize(handle, trans, m, nnzL, descr,
                                   (double *)d_csrVal, d_csrRowPtr, d_csrColInd, info, &pBufferSize);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_bufferSize(handle, trans, m, nnzL, descr,
                                   (float *)d_csrVal, d_csrRowPtr, d_csrColInd, info, &pBufferSize);
    }
        
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis 
    for (int i = 0; i < 1; i++)
    {
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_analysis(handle, trans, m, nnzL, descr, 
                                 (double *)d_csrVal, d_csrRowPtr, d_csrColInd,
                                 info, policy, pBuffer);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_analysis(handle, trans, m, nnzL, descr, 
                                 (float *)d_csrVal, d_csrRowPtr, d_csrColInd,
                                 info, policy, pBuffer);
    }
        
    // L has unit diagonal, so no structural zero is reported.
    status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_cusparse_csrsv2_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cusparse_csrsv2_analysis /= 1;
    *preprocessing_csrsv2 = time_cusparse_csrsv2_analysis;

    printf("cuSPARSE SpTS (csrsv2) analysis on L used %4.2f ms\n", time_cusparse_csrsv2_analysis);

    //  - cuSPARSE SpTS (csrsv2) solve start!
    printf(" - cuSPARSE SpTS (csrsv2) solve start!\n");

    // step 5: solve L*y = x
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_solve(handle, trans, m, nnzL, &alpha_double, descr,
                              (double *)d_csrVal, d_csrRowPtr, d_csrColInd, info,
                              (double *)d_b, (double *)d_x, policy, pBuffer);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_solve(handle, trans, m, nnzL, &alpha_float, descr,
                              (float *)d_csrVal, d_csrRowPtr, d_csrColInd, info,
                              (float *)d_b, (float *)d_x, policy, pBuffer);
    }
        

    // L has unit diagonal, so no numerical zero is reported.
    status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_cusparse_csrsv2_solve = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cusparse_csrsv2_solve /= BENCH_REPEAT;
    *runtime_csrsv2 = time_cusparse_csrsv2_solve;

    printf("cuSPARSE SpTS (csrsv2) solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cusparse_csrsv2_solve, 2*nnzL/(1e6*time_cusparse_csrsv2_solve));
    *gflops_csrsv2 = 2*nnzL/(1e6*time_cusparse_csrsv2_solve);

    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
            err_counter++;
    }

    if (!err_counter)
        printf("cuSPARSE SpTS (csrsv2) on L passed!\n");
    else
        printf("cuSPARSE SpTS (csrsv2) on L failed!\n");

    // step 6: free resources
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_x);

    cudaFree(pBuffer);
    cusparseDestroySolveAnalysisInfo(csrsv_info);
    cusparseDestroyCsrsv2Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    return 0;
}





