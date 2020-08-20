#ifndef _UTILS_SPTRSV_CUDA_
#define _UTILS_SPTRSV_CUDA_

#include "common.h"
#include <cuda_runtime.h>
#include "cusparse.h"

//#define LEVELSET_NLEVEL_THRESHOLD 20
//#define LEVELSET_SHORTROW_THRESHOLD 8
//#define SYNCFREE_SMEM_THRESHOLD 20

__global__
void sptrsv_syncfree_csc_cuda_analyser(const int   *d_cscRowIdx,
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
void sptrsv_syncfree_warpvec_smem_cuda_executor(const int* __restrict__        d_cscColPtr,
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
}*/

/*
__global__
void sptrsv_syncfree_warpvec_smem_csc_cuda_executor(const int*         d_cscColPtr,
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
    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize
    //int global_x_id = 0;
    //if (!lane_id)
    //    global_x_id = atomicAdd(d_id_extractor, 1);
    //global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;
    
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    starting_x = substitution == SUBSTITUTION_FORWARD ? 
                  starting_x : m - 1 - starting_x;
    
    // Prefetch
    const int colstart = d_cscColPtr[global_x_id];
    const int colstop  = d_cscColPtr[global_x_id+1];
    const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

    const int perm_id = substitution == SUBSTITUTION_FORWARD ? 
                        d_levelItem[global_x_id] : d_levelItem[m - 1 - global_x_id];
    const VALUE_TYPE valb = d_b[perm_id];

    if (threadIdx.x < WARP_PER_BLOCK) { s_graphInDegree[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    // Consumer
    do {
        __threadfence_block();
    }
    //while (d_graphInDegree[perm_id] != 1);
    while (s_graphInDegree[local_warp_id] != d_graphInDegree[global_x_id]);

    //VALUE_TYPE xi = d_left_sum[perm_id];
    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
    xi = (valb - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart+1 : colstart;
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? colstop : colstop-1;
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
            //__threadfence_block();
            atomicAdd((int *)&s_graphInDegree[pos], 1);
        }
        else {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            //__threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }

    //finish
    if (!lane_id) d_x[perm_id] = xi;
}
*/

__global__
void sptrsv_syncfree_warpvec_csc_cuda_executor(const int*         d_cscColPtr,
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
					  int* d_levelItem)
{
    // Initialize
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);
    //const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    //int global_x_id = global_id / WARP_SIZE;

    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;
    
    // Prefetch
    const int colstart = d_cscColPtr[global_x_id];
    const int colstop  = d_cscColPtr[global_x_id+1];
    const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

    const int perm_id = global_x_id; //substitution == SUBSTITUTION_FORWARD ? 
                        //d_levelItem[global_x_id] : d_levelItem[m - 1 - global_x_id];
    const VALUE_TYPE valb = d_b[perm_id];

    // Consumer
    do {
        __threadfence_block();
    }
    while (d_graphInDegree[perm_id] != 1);

    VALUE_TYPE xi = d_left_sum[perm_id];
    xi = (valb - xi) * coef;

    //finish
    //if (!lane_id) d_x[perm_id] = xi;


    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart+1 : colstart;
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? colstop : colstop-1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];

	//atomicSub(&d_graphInDegree[rowIdx], 1);
        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        //atomicSub(&d_graphInDegree[rowIdx], 1);
	__threadfence();

	//__threadfence_block();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }
    //__threadfence();

    //finish
    if (!lane_id) d_x[perm_id] = xi;
}
/*
__global__
void sptrsv_syncfree_threadsca_csc_cuda_executor(const int*         d_cscColPtr,
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
    int global_x_id = atomicAdd(d_id_extractor, 1);
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;
    
    // Prefetch
    const int colstart = d_cscColPtr[global_x_id];
    const int colstop  = d_cscColPtr[global_x_id+1];
    const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

    const int perm_id = substitution == SUBSTITUTION_FORWARD ? 
                        d_levelItem[global_x_id] : d_levelItem[m - 1 - global_x_id];
    const VALUE_TYPE valb = d_b[perm_id];

    // Consumer
    do {
        //__threadfence_block();
    }
    while (d_graphInDegree[perm_id] != 1);

    VALUE_TYPE xi = d_left_sum[perm_id];
    xi = (valb - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart+1 : colstart;
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? colstop : colstop-1;
    for (int jj = start_ptr; jj < stop_ptr; jj++)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];

        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        //__threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }

    //finish
    d_x[perm_id] = xi;
}
*/

/*
__global__
void sptrsv_levelset_threadsca_csc_cuda_executor(const int*         d_cscColPtr,
                                          const int*         d_cscRowIdx,
                                          const VALUE_TYPE*  d_cscVal,
                                          VALUE_TYPE*                    d_left_sum,
                                          const int                      m,
                                          const int                      offset,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x,
                                          int *d_levelItem)
{

    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        const int colidx = global_x_id+offset;

        const int colstart = d_cscColPtr[colidx];
        const int colstop  = d_cscColPtr[colidx+1];
        const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop-1;
        const int perm_id = substitution == SUBSTITUTION_FORWARD ? 
                        d_levelItem[colidx] : d_levelItem[m - 1 - colidx];

        VALUE_TYPE xi = d_left_sum[perm_id];
        xi = (d_b[perm_id] - xi) / d_cscVal[pos];

        const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart+1 : colstart;
        const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? colstop : colstop-1;
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];

            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        }
        d_x[perm_id] = xi;
    }
}
*/

__global__
void sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack(const int*         d_csrRowPtr,
                                          const int*         d_csrColIdx,
                                          const VALUE_TYPE*  d_csrVal,
                                          const int                      m,
                                          const int                      offset,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        const int rowidx = global_x_id+offset;
        //const int perm_id = rowidx;//d_levelItem[rowidx];
        //d_x[perm_id] = d_b[perm_id] / d_csrVal[d_csrRowPtr[rowidx+1] - 1];
        d_x[rowidx] = d_b[rowidx] / d_csrVal[d_csrRowPtr[rowidx+1] - 1];
    }
}

/*
__global__
void sptrsv_cusparse_perm_cuda_executor(const int                      m,
                                              VALUE_TYPE *d_x,
                                        const VALUE_TYPE*                    d_x_perm)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        const int rowidx = global_x_id;
        const int perm_id = rowidx; //d_levelItem[rowidx];
        d_x[perm_id] = d_x_perm[rowidx];
    }
}
*/


__global__
void sptrsv_levelset_threadsca_csr_cuda_executor(const int*         d_csrRowPtr,
                                          const int*         d_csrColIdx,
                                          const VALUE_TYPE*  d_csrVal,
                                          const int                      m,
                                          const int                      offset,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        const int rowidx = global_x_id+offset;
        const int start = d_csrRowPtr[rowidx];
        const int stop  = d_csrRowPtr[rowidx+1];
        VALUE_TYPE sum = 0;
        for (int j = start; j < stop - 1; j++)
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];

        //const int perm_id = rowidx; //d_levelItem[rowidx];
        //d_x[perm_id] = (d_b[perm_id] - sum) / d_csrVal[stop - 1];
        d_x[rowidx] = (d_b[rowidx] - sum) / d_csrVal[stop - 1];
    }
}

__global__
void sptrsv_levelset_warpvec_csr_cuda_executor(const int*         d_csrRowPtr,
                                          const int*         d_csrColIdx,
                                          const VALUE_TYPE*  d_csrVal,
                                          const int                      m,
                                          const int                      offset,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    int rowidx = global_id / WARP_SIZE;
    if (rowidx >= m) return;

    rowidx += offset;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = d_csrRowPtr[rowidx];
    const int stop  = d_csrRowPtr[rowidx+1];

    VALUE_TYPE sum = 0;
    for (int j = start + lane_id; j < stop - 1; j += WARP_SIZE)
    {
        sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
    }
    sum = sum_32_shfl(sum);

    //finish
    if (!lane_id) 
    {
        //const int perm_id = rowidx; //d_levelItem[rowidx];
        //d_x[perm_id] = (d_b[perm_id] - sum) / d_csrVal[stop - 1];
        d_x[rowidx] = (d_b[rowidx] - sum) / d_csrVal[stop - 1];
    }
}

__global__
void sptrsv_syncfree_csc_cuda_executor_fasttrack(const int*         d_cscColPtr,
                                          const int*         d_cscRowIdx,
                                          const VALUE_TYPE*  d_cscVal,
                                          const int                      m,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        //const int pos = substitution == SUBSTITUTION_FORWARD ? 
        //                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
        //d_x[global_x_id] = d_b[global_x_id] / d_cscVal[pos];

        d_x[global_x_id] = d_b[global_x_id] / d_cscVal[global_x_id];
    }
}

int sptrsv_syncfree_csc_cuda(const int           *cscColPtrTR,
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
                                    int           *nnzsum,
                                    double        *timesum,
                                    double        *preprocess)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }
    struct timeval t1, t2;

    const int fasttrack = m == nnzTR ? 1 : 0;

    //if (m == nnzTR)
    //    printf("m == nnzTR, SpTRSV could be removed\n");
    int nlv = 0;
    int *levelItem  = (int *)malloc(m * sizeof(int));
    int *levelPtr = (int *)malloc((m+1) * sizeof(int));
    int *cscColPtrTR_new = (int *)malloc((n+1) * sizeof(int));
    int *cscRowIdxTR_new = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *cscValTR_new = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));
    int *csrRowPtrTR = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxTR = (int *)malloc(nnzTR * sizeof(int));
    VALUE_TYPE *csrValTR = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));
    //int *csrRowPtrTR_new = (int *)malloc((m+1) * sizeof(int));
    //int *csrColIdxTR_new = (int *)malloc(nnzTR * sizeof(int));
    //VALUE_TYPE *csrValTR_new = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

    gettimeofday(&t1, NULL);
    matrix_transposition(n, m, nnzTR,
                             cscColPtrTR, cscRowIdxTR, cscValTR,
                             csrColIdxTR, csrRowPtrTR, csrValTR);

    if (fasttrack) // fast track
    {
        nlv = 1;
        memcpy(cscColPtrTR_new, cscColPtrTR, (n+1) * sizeof(int));
        memcpy(cscRowIdxTR_new, cscRowIdxTR, nnzTR * sizeof(int));
        memcpy(cscValTR_new, cscValTR, nnzTR * sizeof(VALUE_TYPE));
    }
    else
    {
        levelset_reordering_col_csc(cscColPtrTR, cscRowIdxTR, cscValTR, 
                            cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new, 
                            levelPtr, levelItem, &nlv, m, n, nnzTR, substitution);

        //levelset_reordering_row_csr(csrRowPtrTR, csrColIdxTR, csrValTR,
        //                       csrRowPtrTR_new, csrColIdxTR_new, csrValTR_new,
        //                       levelPtr, levelItem, &nlv, m, n, nnzTR, substitution);
    }
    //printf ("tri level: nlv = %i, m = %i, nnz = %i, nnz/col = %4.2f\n", nlv, m, nnzTR, (double)nnzTR/(double)m);
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;



    // transfer host mem to device mem
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    int *d_csrRowPtrTR;
    int *d_csrColIdxTR;
    VALUE_TYPE *d_csrValTR;

    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrTR, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR  * sizeof(int));
    cudaMalloc((void **)&d_cscValTR,    nnzTR  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, nnzTR  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR,    cscValTR,    nnzTR  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_csrRowPtrTR, (m+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxTR, nnzTR  * sizeof(int));
    cudaMalloc((void **)&d_csrValTR,    nnzTR  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtrTR, csrRowPtrTR, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxTR, csrColIdxTR, nnzTR  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValTR,    csrValTR,    nnzTR  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

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
    
    //  - cuda reblocking SpTRSV analysis start!
    //printf(" - cuda reblocking SpTRSV (level 1) analysis start!\n");

    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));

    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));

    int num_threads = 128;
    int num_blocks = ceil ((double)nnzTR / (double)num_threads);

    gettimeofday(&t1, NULL);
    for (int i = 0; i < 1; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsv_syncfree_csc_cuda_analyser<<< num_blocks, num_threads >>>
                                      (d_cscRowIdxTR, m, nnzTR, d_graphInDegree);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= 1;
    *preprocess += time_cuda_analysis;

    //printf("cuda reblocking SpTRSV (level 1) analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - cuda reblocking SpTRSV solve start!
    //printf(" - cuda reblocking SpTRSV (level 1) solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m * rhs);

    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);

    // this is for profiling while loop only
    int *d_while_profiler;
    //cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    //cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    //int *while_profiler = (int *)malloc(sizeof(int) * n);


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
        // step 2: create a empty info structure
        cusparseCreateCsrsv2Info(&info);

        // step 3: query how much memory used in csrsv2, and allocate the buffer
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_bufferSize(handle, trans, m, nnzTR, descr,
                                   (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info, &pBufferSize);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_bufferSize(handle, trans, m, nnzTR, descr,
                                   (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info, &pBufferSize);

        // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
        cudaMalloc((void**)&pBuffer, pBufferSize);


        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_analysis(handle, trans, m, nnzTR, descr,
                                 (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                 info, policy, pBuffer);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_analysis(handle, trans, m, nnzTR, descr,
                                 (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                 info, policy, pBuffer);

        // L has unit diagonal, so no structural zero is reported.
        status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status){
            printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
        }
        VALUE_TYPE *d_x_perm;
        cudaMalloc((void **)&d_x_perm, n  * sizeof(VALUE_TYPE));
        cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));














    // step 5: solve L*y = x
    double time_cuda_solve = 0;
    int method = -1;
    int repeat_real = -1;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        if (!fasttrack)
        {
            // get a unmodified in-degree array, only for benchmarking use
            cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
            //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);
            
            // clear left_sum array, only for benchmarking use
            cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);
            cudaMemset(d_id_extractor, 0, sizeof(int));
            cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
	}


        gettimeofday(&t1, NULL);

        //if (rhs == 1)
        {
            if (fasttrack)
            {
                //printf("ONLY DIA (m = %i): if (fasttrack)\n", m);
                num_threads = WARP_PER_BLOCK * WARP_SIZE;
                num_blocks = ceil ((double)m / (double)num_threads);
                sptrsv_syncfree_csc_cuda_executor_fasttrack<<< num_blocks, num_threads >>>
                                         (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                          m, substitution, d_b, d_x);
            }
            else
            {
		int nnzr = nnzTR / m;
                if (nlv > 20000)
		{
	            //printf("CUSPARSE (m = %i): if (nlv > 20000) nnzTR = %i, nnzr = %i, nlevel = %i\n", m, nnzTR, nnzr, nlv);
		    method = 1;
		    if (sizeof(VALUE_TYPE) == 8)
                        cusparseDcsrsv2_solve(handle, trans, m, nnzTR, &alpha_double, descr,
                              (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info,
                              (double *)d_b, (double *)d_x, policy, pBuffer);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_solve(handle, trans, m, nnzTR, &alpha_float, descr,
                              (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info,
                              (float *)d_b, (float *)d_x, policy, pBuffer);

		}
	        else if ((nnzr <= 15 && nlv <= 20) || (nnzr == 1 && nlv <= 100))
                {
                    //printf("LEVEL-SET (m = %i): else if ((nnzr <= 15 && nlv <= 13) || (nnzr == 1 && nlv <= 100)), nnzTR = %i, nnzr = %i, nlevel = %i\n", 
                    //       m, nnzTR, nnzr, nlv);
		    method = 0;
                    for (int li = 0; li < nlv; li++)
                    {
                        const int m_lv = levelPtr[li+1] - levelPtr[li];
                        //printf("li = %i, m_lv = %i\n", li, m_lv);
                        const int offset = levelPtr[li];
                        int nnz_lv = 0;
                        for (int lvi = levelPtr[li]; lvi < levelPtr[li+1]; lvi++)
                        {
                            nnz_lv += csrRowPtrTR[lvi+1] - csrRowPtrTR[lvi];
                        }
                        //printf("li = %i, m_lv = %i, nnz_lv = %i, nnz_lv/m_lv = %4.2f\n", li, m_lv, nnz_lv, (double)nnz_lv/(double)m_lv);

                        
                        if (li == 0)
                        {
                            num_threads = WARP_PER_BLOCK * WARP_SIZE;
                            num_blocks = ceil ((double)m_lv / (double)num_threads);
                            sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack<<< num_blocks, num_threads >>>
                                                (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                                m_lv, offset, substitution, d_b, d_x);
                        }
                        else
                        {
                            if ((nnz_lv / m_lv) <= 15)
                            {
                                num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                num_blocks = ceil ((double)m_lv / (double)num_threads);
                                sptrsv_levelset_threadsca_csr_cuda_executor<<< num_blocks, num_threads >>>
                                            (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                            m_lv, offset, substitution, d_b, d_x);
                            }
                            else
                            {
                                num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                num_blocks = ceil ((double)m_lv / (double)(num_threads/WARP_SIZE));
                                sptrsv_levelset_warpvec_csr_cuda_executor<<< num_blocks, num_threads >>>
                                            (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                            m_lv, offset, substitution, d_b, d_x);
                            }
                        }
                        
                        //sptrsv_levelset_threadsca_csc_cuda_executor<<< num_blocks, num_threads >>>
                        //                      (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                        //                       d_left_sum, m_lv, offset, substitution, d_b, d_x, d_levelItem);
                    }
                }
                else
                {
                    //printf("SYNC-FREE (m = %i): else, nlevel = %i, nnzTR = %i, nnzr = %i, nlv = %i\n", 
                    //       m, nlv, nnzTR, nnzr, nlv);
		    method = 2;
                    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
                    sptrsv_syncfree_warpvec_csc_cuda_executor<<< num_blocks, num_threads >>>
                                            (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                            d_graphInDegree, d_left_sum,
                                            m, substitution, d_b, d_x, d_while_profiler, 
                                            d_id_extractor, d_levelItem);
                }
            }
        }
        //else
        //{
            //num_threads = 4 * WARP_SIZE;
            //num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            //sptrsm_syncfree_reorder_cuda_executor_update<<< num_blocks, num_threads >>>
            //                             (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
            //                              d_graphInDegree, d_left_sum,
            //                              m, substitution, rhs, opt,
            //                              d_b, d_x, d_while_profiler, d_id_extractor);
        //}

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        double time_onerun = 0;
        if (i == 5)
            time_onerun = time_cuda_solve / (i+1);
        if (time_onerun > 10.0) // if longer than 10 ms
        {
            repeat_real = i+1;
            break;
        }
    }

    time_cuda_solve /= (repeat_real == -1) ? BENCH_REPEAT : repeat_real;
    double flop = 2*(double)rhs*(double)nnzTR;
    //printf("SpTRSV nnzr = %i, nlv = %i, used method = %i, %4.4f ms\n", nnzTR/m, nlv, method, time_cuda_solve);
    //printf("cuda reblocking SpTRSV (level 1) solve used %4.2f ms, throughput is %4.2f gflops, ",
    //       time_cuda_solve, flop/(1e6*time_cuda_solve));
    //*gflops = flop/(1e6*time_cuda_solve);

    cudaMemcpy(x, d_x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    /*double accuracy = 1e-4;
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
        printf("check passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("check _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);*/

    // profile while loop
    //cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    //long long unsigned int while_count = 0;
    //for (int i = 0; i < n; i++)
    //{
    //    while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    //}
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    //free(while_profiler);











/*
    if (!fasttrack)
    {
	    int *d_placeholder;
        int BENCH_REPEAT_KERNEL = 5;

        if (time_cuda_solve >= 50)
            BENCH_REPEAT_KERNEL = 3;
        else if (time_cuda_solve >= 5 && time_cuda_solve < 50)
            BENCH_REPEAT_KERNEL = 10;
        else
            BENCH_REPEAT_KERNEL = 100;

        // test kernel
        double time_cuda_solve_syncfree = 0;
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            // get a unmodified in-degree array, only for benchmarking use
            cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
            //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);
            
            // clear left_sum array, only for benchmarking use
            cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);
            cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
            cudaMemset(d_id_extractor, 0, sizeof(int));

            gettimeofday(&t1, NULL);

            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            sptrsv_syncfree_warpvec_csc_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                    d_graphInDegree, d_left_sum,
                                    m, substitution, d_b, d_x, d_while_profiler, 
                                    d_id_extractor, d_placeholder);

            cudaDeviceSynchronize();
            gettimeofday(&t2, NULL);

            time_cuda_solve_syncfree += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }
        time_cuda_solve_syncfree /= BENCH_REPEAT_KERNEL;

        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);


        double time_cuda_solve_levelset_sca = 0;
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
            //printf("LEVEL-SET (m = %i): else if (!fasttrack && nlv <= LEVELSET_NLEVEL_THRESHOLD), nnzTR = %i, nlevel = %i\n", 
            //       m, nnzTR, nlv);
            gettimeofday(&t1, NULL);
            for (int li = 0; li < nlv; li++)
            {
                const int m_lv = levelPtr[li+1] - levelPtr[li];
                //printf("li = %i, m_lv = %i\n", li, m_lv);
                const int offset = levelPtr[li];
                int nnz_lv = 0;
                for (int lvi = levelPtr[li]; lvi < levelPtr[li+1]; lvi++)
                {
                    nnz_lv += csrRowPtrTR_new[lvi+1] - csrRowPtrTR_new[lvi];
                }
                //printf("li = %i, m_lv = %i, nnz_lv = %i, nnz_lv/m_lv = %4.2f\n", li, m_lv, nnz_lv, (double)nnz_lv/(double)m_lv);

                if (li == 0)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil ((double)m_lv / (double)num_threads);
                    sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack<<< num_blocks, num_threads >>>
                                        (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                        m_lv, offset, substitution, d_b, d_x, d_levelItem);
                }
                else
                {
                    //if (nnz_lv / m_lv < LEVELSET_SHORTROW_THRESHOLD)
                    //{
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil ((double)m_lv / (double)num_threads);
                        sptrsv_levelset_threadsca_csr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                    m_lv, offset, substitution, d_b, d_x, d_levelItem);
                    //}
                    //else
                    //{
                    //    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    //    num_blocks = ceil ((double)m_lv / (double)(num_threads/WARP_SIZE));
                    //    sptrsv_levelset_warpvec_csr_cuda_executor<<< num_blocks, num_threads >>>
                    //                (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                    //                m_lv, offset, substitution, d_b, d_x, d_levelItem);
                    //}
                }
            }

            cudaDeviceSynchronize();
            gettimeofday(&t2, NULL);

            time_cuda_solve_levelset_sca += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }

        time_cuda_solve_levelset_sca /= BENCH_REPEAT_KERNEL;


        double time_cuda_solve_levelset = 0;
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
            //printf("LEVEL-SET (m = %i): else if (!fasttrack && nlv <= LEVELSET_NLEVEL_THRESHOLD), nnzTR = %i, nlevel = %i\n", 
            //       m, nnzTR, nlv);
            gettimeofday(&t1, NULL);
            for (int li = 0; li < nlv; li++)
            {
                const int m_lv = levelPtr[li+1] - levelPtr[li];
                //printf("li = %i, m_lv = %i\n", li, m_lv);
                const int offset = levelPtr[li];
                int nnz_lv = 0;
                for (int lvi = levelPtr[li]; lvi < levelPtr[li+1]; lvi++)
                {
                    nnz_lv += csrRowPtrTR_new[lvi+1] - csrRowPtrTR_new[lvi];
                }
                //printf("li = %i, m_lv = %i, nnz_lv = %i, nnz_lv/m_lv = %4.2f\n", li, m_lv, nnz_lv, (double)nnz_lv/(double)m_lv);

                if (li == 0)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil ((double)m_lv / (double)num_threads);
                    sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack<<< num_blocks, num_threads >>>
                                        (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                        m_lv, offset, substitution, d_b, d_x, d_placeholder);
                }
                else
                {
                    if (nnz_lv / m_lv <= 8)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil ((double)m_lv / (double)num_threads);
                        sptrsv_levelset_threadsca_csr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                    m_lv, offset, substitution, d_b, d_x, d_placeholder);
                    }
                    else
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil ((double)m_lv / (double)(num_threads/WARP_SIZE));
                        sptrsv_levelset_warpvec_csr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR,
                                    m_lv, offset, substitution, d_b, d_x, d_placeholder);
                    }
                }
            }
            cudaDeviceSynchronize();
            gettimeofday(&t2, NULL);

            time_cuda_solve_levelset += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }

        time_cuda_solve_levelset /= BENCH_REPEAT_KERNEL;



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
        // step 2: create a empty info structure
        cusparseCreateCsrsv2Info(&info);

        // step 3: query how much memory used in csrsv2, and allocate the buffer
        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_bufferSize(handle, trans, m, nnzTR, descr,
                                   (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info, &pBufferSize);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_bufferSize(handle, trans, m, nnzTR, descr,
                                   (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info, &pBufferSize);

        // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
        cudaMalloc((void**)&pBuffer, pBufferSize);


        if (sizeof(VALUE_TYPE) == 8)
            cusparseDcsrsv2_analysis(handle, trans, m, nnzTR, descr,
                                 (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                 info, policy, pBuffer);
        else if (sizeof(VALUE_TYPE) == 4)
            cusparseScsrsv2_analysis(handle, trans, m, nnzTR, descr,
                                 (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                 info, policy, pBuffer);

        // L has unit diagonal, so no structural zero is reported.
        status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status){
            printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
        }
	VALUE_TYPE *d_x_perm;
	cudaMalloc((void **)&d_x_perm, n  * sizeof(VALUE_TYPE));
        cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));



        double time_cuda_solve_cusparse_v2 = 0;
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            cudaMemset(d_x_perm, 0, sizeof(VALUE_TYPE) * n * rhs);
            //printf("LEVEL-SET (m = %i): else if (!fasttrack && nlv <= LEVELSET_NLEVEL_THRESHOLD), nnzTR = %i, nlevel = %i\n", 
            //       m, nnzTR, nlv);
            gettimeofday(&t1, NULL);


            if (sizeof(VALUE_TYPE) == 8)
                cusparseDcsrsv2_solve(handle, trans, m, nnzTR, &alpha_double, descr,
                              (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info,
                              (double *)d_b, (double *)d_x, policy, pBuffer);
            else if (sizeof(VALUE_TYPE) == 4)
                cusparseScsrsv2_solve(handle, trans, m, nnzTR, &alpha_float, descr,
                              (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, info,
                              (float *)d_b, (float *)d_x, policy, pBuffer);

            // permute x
	    //int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            //int num_blocks = ceil ((double)m / (double)num_threads);
	    //sptrsv_cusparse_perm_cuda_executor<<< num_blocks, num_threads >>>(m, d_x, d_x_perm, d_levelItem);

	    cudaDeviceSynchronize();
            gettimeofday(&t2, NULL);


            // L has unit diagonal, so no numerical zero is reported.
            status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
            if (CUSPARSE_STATUS_ZERO_PIVOT == status){
                printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
            }

            time_cuda_solve_cusparse_v2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }

        time_cuda_solve_cusparse_v2 /= BENCH_REPEAT_KERNEL;

        cudaFree(d_x_perm);
        cudaFree(pBuffer);
        cusparseDestroySolveAnalysisInfo(csrsv_info);
        cusparseDestroyCsrsv2Info(info);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);


        int nnzr = nnzTR / m;

        double time_sptrsv[3];
        time_sptrsv[0] = time_cuda_solve_levelset;
        time_sptrsv[1] = time_cuda_solve_cusparse_v2; //time_cuda_solve_levelset_vec;
        time_sptrsv[2] = time_cuda_solve_syncfree;

        int best_idx[3] = {0, 1, 2};
        quicksort_keyval<double, int>(time_sptrsv, best_idx, 0, 3-1);
        int selected = best_idx[0];

        //double time_cuda_solve_levelset_best = time_cuda_solve_levelset_sca < time_cuda_solve_levelset_vec ? time_cuda_solve_levelset_sca : time_cuda_solve_levelset_vec;
        
        //int selected = time_cuda_solve_syncfree < time_cuda_solve_levelset_best ? 3 : -1;
        //double error = abs(time_cuda_solve_syncfree - time_cuda_solve_levelset_best);
        //double rate = error / (time_cuda_solve_syncfree > time_cuda_solve_levelset_best ? time_cuda_solve_levelset_best : time_cuda_solve_syncfree);
        //selected = rate < 0.03 ? 4 : selected;

        //if (selected == -1)
        //{
        //    selected = time_cuda_solve_levelset_sca < time_cuda_solve_levelset_vec ? 0 : 1;
        //    double error = abs(time_cuda_solve_levelset_vec - time_cuda_solve_levelset_sca);
        //    double rate = error / (time_cuda_solve_levelset_vec > time_cuda_solve_levelset_sca ? time_cuda_solve_levelset_sca : time_cuda_solve_levelset_vec);
        //    selected = rate < 0.03 ? 2 : selected;
        //}

        FILE *fout;
        if (sizeof(VALUE_TYPE) == 8)
            fout = fopen("results-sptrsv-dp-kernel-nnzr.csv", "a");
        else
            fout = fopen("results-sptrsv-sp-kernel-nnzr.csv", "a");

        if (fout == NULL) printf("Writing results fails.\n");
        fprintf(fout, "%i,%i,%f,%f,%f,%i,%i,%i\n", 
                m, nnzTR, 
                time_cuda_solve_levelset, time_cuda_solve_cusparse_v2, time_cuda_solve_syncfree, 
                nnzr, nlv, selected);
        fclose(fout);
    }
*/








    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(d_id_extractor);
    cudaFree(d_left_sum);
    //cudaFree(d_while_profiler);

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_csrRowPtrTR);
    cudaFree(d_csrColIdxTR);
    cudaFree(d_csrValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    
    free(cscColPtrTR_new);
    free(cscRowIdxTR_new);
    free(cscValTR_new);
    //free(csrRowPtrTR_new);
    //free(csrColIdxTR_new);
    //free(csrValTR_new);

    //cudaFree(d_levelItem);
    //free(levelPtr);
    //free(levelItem);

    *nnzsum += nnzTR;
    *timesum += time_cuda_solve;

    return 0;
}

#endif
