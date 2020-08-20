#ifndef _SPTRSV_SYNCFREE_REORDER_CUDA_
#define _SPTRSV_SYNCFREE_REORDER_CUDA_

#include "common.h"
#include "utils.h"
#include "findlevel.h"
#include <cuda_runtime.h>

__global__
void sptrsv_syncfree_reorder_cuda_analyser(const int   *d_cscRowIdx,
                                   const int    m,
                                   const int    nnz,
                                         int   *d_graphInDegree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}

/*
__global__
void sptrsv_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     d_graphInDegree,
                                         VALUE_TYPE*              d_left_sum,
                                   const int                      m,
                                   const int                      substitution,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     d_while_profiler)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    starting_x = substitution == SUBSTITUTION_FORWARD ? 
                  starting_x : m - 1 - starting_x;
    
    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                    d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK) { s_graphInDegree[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (s_graphInDegree[local_warp_id] != d_graphInDegree[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    //}
    //while (s_graphInDegree[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];
        const bool cond = substitution == SUBSTITUTION_FORWARD ? 
                    (rowIdx < starting_x + WARP_PER_BLOCK) : (rowIdx > starting_x - WARP_PER_BLOCK);
        if (cond) {
            const int pos = substitution == SUBSTITUTION_FORWARD ? 
                            rowIdx - starting_x : starting_x - rowIdx;
            atomicAdd((VALUE_TYPE *)&s_left_sum[pos], xi * d_cscVal[j]);
            __threadfence_block();
            atomicAdd((int *)&s_graphInDegree[pos], 1);
        }
        else {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }

    //finish
    if (!lane_id) d_x[global_x_id] = xi;
}
*/

__global__
void sptrsv_syncfree_reorder_cuda_executor_update(const int*         d_cscColPtr,
                                          const int*         d_cscRowIdx,
                                          const VALUE_TYPE*  d_cscVal,
                                          int*                           d_graphInDegree,
                                          VALUE_TYPE*                    d_left_sum,
                                          const int                      m,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x,
                                          int*                           d_while_profiler,
                                          int*                           d_id_extractor,
                                          int *d_levelItem)
{
    //const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    //const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;
    
    // Prefetch
    const int colstart = d_cscColPtr[global_x_id];
    const int colstop  = d_cscColPtr[global_x_id+1];
    const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    //const int check_id = substitution == SUBSTITUTION_FORWARD ? 
    //                     d_cscRowIdx[colstart] : d_cscRowIdx[colstop-1];
    const int perm_id = substitution == SUBSTITUTION_FORWARD ? 
                        d_levelItem[global_x_id] : d_levelItem[m - 1 - global_x_id];
    //if (!lane_id) printf("global_x_id = %i, ti = %i, check_id = %i\n", global_x_id, ti, check_id);

    // Consumer
    do {
        __threadfence_block();
    }
    //while (d_graphInDegree[global_x_id] != 1);
    while (d_graphInDegree[perm_id] != 1);

    //VALUE_TYPE xi = d_left_sum[global_x_id];
    VALUE_TYPE xi = d_left_sum[perm_id];
    //xi = (d_b[global_x_id] - xi) * coef;
    xi = (d_b[perm_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart+1 : colstart;
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? colstop : colstop-1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];

        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        __threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }

    //finish
    //if (!lane_id) d_x[global_x_id] = xi;
    if (!lane_id) d_x[perm_id] = xi;
}

/*
__global__
void sptrsm_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     d_graphInDegree,
                                         VALUE_TYPE*              d_left_sum,
                                   const int                      m,
                                   const int                      substitution,
                                   const int                      rhs,
                                   const int                      opt,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     d_while_profiler)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    // Initialize
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (1 != d_graphInDegree[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    //}
    //while (1 != graphInDegree );

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;

    if (opt == OPT_WARP_NNZ)
    {
        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = 0; k < rhs; k++)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_RHS)
    {
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_AUTO)
    {
        const int len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 16) && len < 2048)
        {
            for (int jj = start_ptr; jj < stop_ptr; jj++)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = 0; k < rhs; k++)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}
*/

/*
__global__
void sptrsm_syncfree_reorder_cuda_executor_update(const int* __restrict__        d_cscColPtr,
                                          const int* __restrict__        d_cscRowIdx,
                                          const VALUE_TYPE* __restrict__ d_cscVal,
                                          int*                           d_graphInDegree,
                                          VALUE_TYPE*                    d_left_sum,
                                          const int                      m,
                                          const int                      substitution,
                                          const int                      rhs,
                                          const int                      opt,
                                          const VALUE_TYPE* __restrict__ d_b,
                                          VALUE_TYPE*                    d_x,
                                          int*                           d_while_profiler,
                                          int*                           d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl(global_x_id, 0);

    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    // Consumer
    do {
        __threadfence_block();
    }
    while (1 != d_graphInDegree[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    //}
    //while (1 != graphInDegree );

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;

    if (opt == OPT_WARP_NNZ)
    {
        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = 0; k < rhs; k++)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_RHS)
    {
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_AUTO)
    {
        const int len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 16) && len < 2048)
        {
            for (int jj = start_ptr; jj < stop_ptr; jj++)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = 0; k < rhs; k++)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}
*/

int sptrsv_syncfree_reorder_cuda(const int           *cscColPtrTR,
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
                               double        *gflops)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }


    // ------------------------
    // code below for for reordering csr according to level-set execution order
    // ------------------------

    // transpose to have csr data
    int *csrRowPtrTR = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxTR = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *csrValTR = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));
    
    // transpose from csc to csr
    matrix_transposition(m, n, nnzTR,
                         cscColPtrTR, cscRowIdxTR, cscValTR,
                         csrColIdxTR, csrRowPtrTR, csrValTR);

    int  *levelPtr  = (int *)malloc((m+1) * sizeof(int));
    int  *levelItem = (int *)malloc(m * sizeof(int));

    int nlv = 0;
    findlevel(cscColPtrTR, cscRowIdxTR, csrRowPtrTR, m, &nlv, levelPtr, levelItem);

    // reorder nonzeros
    //int *csrRowPtrTR_new = (int *)malloc((m+1) * sizeof(int));
    //csrRowPtrTR_new[0] = 0;
    //int *csrColIdxTR_new = (int *)malloc(nnzTR * sizeof(int));
    //VALUE_TYPE *csrValTR_new = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));
    //VALUE_TYPE *dia_new = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    /*for (int i = 0; i < n; i++)
    {
        int idx = levelItem[i];
        int nnzr = csrRowPtrTR[idx+1] - csrRowPtrTR[idx]; 
        csrRowPtrTR_new[i+1] = csrRowPtrTR_new[i] + nnzr;

        for (int j = 0; j < nnzr; j++)
        {
            int off = csrRowPtrTR[idx] + j;
            int off_new = csrRowPtrTR_new[i] + j;
            //if (csrColIdxTR[off] != idx)
            //{
                csrColIdxTR_new[off_new] = csrColIdxTR[off];
                csrValTR_new[off_new] = csrValTR[off];
            //}
            //else
            //{
            //    dia_new[i] = csrVal[off];
            //}
        }
    }*/


    int *cscColPtrTR_new = (int *)malloc((n+1) * sizeof(int));
    cscColPtrTR_new[0] = 0;
    int *cscRowIdxTR_new = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *cscValTR_new = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

    for (int i = 0; i < n; i++)
    {
        int idx = substitution == SUBSTITUTION_FORWARD ? levelItem[i] : levelItem[n - i - 1];
        int nnzr = cscColPtrTR[idx+1] - cscColPtrTR[idx]; 
        cscColPtrTR_new[i+1] = cscColPtrTR_new[i] + nnzr;

        for (int j = 0; j < nnzr; j++)
        {
            int off = cscColPtrTR[idx] + j;
            int off_new = cscColPtrTR_new[i] + j;
            //if (csrColIdxTR[off] != idx)
            //{
                cscRowIdxTR_new[off_new] = cscRowIdxTR[off];
                cscValTR_new[off_new] = cscValTR[off];
            //}
            //else
            //{
            //    dia_new[i] = csrVal[off];
            //}
        }
    }

    /*struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // level-set SpTRSV in parallel 
        for (int li = 0; li < nlv; li++)
        {
            #pragma omp parallel for
            for (int ri = levelPtr[li]; ri < levelPtr[li+1]; ri++)
            {
                VALUE_TYPE sum = 0;
                for (int j = csrRowPtr_new[ri]; j < csrRowPtr_new[ri+1]; j++)
                {
                    sum += x[csrColIdx_new[j]] * csrVal_new[j];
                }
                int ti = levelItem[ri];
                x[ti] = (b[ti] - sum) / dia_new[ri];
            }
        }
    }

    gettimeofday(&t2, NULL);
    double time_sptrsv_levelset = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("\n\nSpTRSV level-set parallel executor used %4.2f ms\n", time_sptrsv_levelset/BENCH_REPEAT);

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n; i++)
    {
        ref += abs(xref[i]);
        res += abs(x[i] - xref[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("SpTRSV level-set parallel executor passed! |x-xref|/|xref| = %8.2e\n\n", res);
    else
        printf("SpTRSV level-set parallel executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n\n", res);*/



    // transpose from csr to csc
    //matrix_transposition(m, n, nnzTR,
    //                     csrRowPtrTR_new, csrColIdxTR_new, csrValTR_new,
    //                     cscRowIdxTR_new, cscColPtrTR_new, cscValTR_new);

    // keep each column sort 
    //for (int i = 0; i < n; i++)
    //{
    //    quick_sort_key_val_pair<int, VALUE_TYPE>(&cscRowIdxTR_new[cscColPtrTR_new[i]],
    //                                          &cscValTR_new[cscColPtrTR_new[i]],
    //                                          cscColPtrTR_new[i+1]-cscColPtrTR_new[i]);
    //}

    //free(csrRowPtrTR_new);
    //free(csrColIdxTR_new);
    //free(csrValTR_new);
    free(csrRowPtrTR);
    free(csrColIdxTR);
    free(csrValTR);
    //free(levelPtr);
    //free(dia_new);
    //free(xref);
    //free(x);
    //free(b);


    // ------------------------
    // code above for reordering csr according to level-set execution order
    // ------------------------

    // transfer host mem to device mem
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrTR, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR  * sizeof(int));
    cudaMalloc((void **)&d_cscValTR,    nnzTR  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrTR, cscColPtrTR_new, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR_new, nnzTR  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR,    cscValTR_new,    nnzTR  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * rhs * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * rhs * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * rhs * sizeof(VALUE_TYPE));

    // permutation info
    int *d_levelItem;
    cudaMalloc((void **)&d_levelItem, m * sizeof(int));
    cudaMemcpy(d_levelItem, levelItem, m * sizeof(int), cudaMemcpyHostToDevice);

    //  - cuda syncfree SpTRSV analysis start!
    printf(" - cuda syncfree SpTRSV (level-set order) analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));

    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));

    int num_threads = 128;
    int num_blocks = ceil ((double)nnzTR / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsv_syncfree_reorder_cuda_analyser<<< num_blocks, num_threads >>>
                                      (d_cscRowIdxTR, m, nnzTR, d_graphInDegree);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("cuda syncfree SpTRSV (level-set order) analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda syncfree SpTRSV (level-set order) solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m * rhs);

    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);

    // this is for profiling while loop only
    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    double time_cuda_solve = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // get a unmodified in-degree array, only for benchmarking use
        cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);
        
        // clear left_sum array, only for benchmarking use
        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);
        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
        cudaMemset(d_id_extractor, 0, sizeof(int));

        gettimeofday(&t1, NULL);

        if (rhs == 1)
        {
            num_threads = WARP_PER_BLOCK * WARP_SIZE;
            //num_threads = 1 * WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            //sptrsv_syncfree_cuda_executor<<< num_blocks, num_threads >>>
            sptrsv_syncfree_reorder_cuda_executor_update<<< num_blocks, num_threads >>>
                                         (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                          d_graphInDegree, d_left_sum,
                                          m, substitution, d_b, d_x, d_while_profiler, 
                                          d_id_extractor, d_levelItem);
        }
        else
        {
            num_threads = 4 * WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            //sptrsm_syncfree_reorder_cuda_executor_update<<< num_blocks, num_threads >>>
            //                             (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
            //                              d_graphInDegree, d_left_sum,
            //                              m, substitution, rhs, opt,
            //                              d_b, d_x, d_while_profiler, d_id_extractor);
        }

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnzTR;

    printf("cuda syncfree SpTRSV (level-set order) solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop/(1e6*time_cuda_solve));
    *gflops = flop/(1e6*time_cuda_solve);

    cudaMemcpy(x, d_x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

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
        printf("cuda syncfree SpTRSV (level-set order) executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("cuda syncfree SpTRSV (level-set order) executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    // profile while loop
    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    free(while_profiler);

    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(d_id_extractor);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_levelItem);

    free(cscColPtrTR_new);
    free(cscRowIdxTR_new);
    free(cscValTR_new);

    free(levelItem);

    return 0;
}

#endif



