#ifndef _UTILS_SPMV_CUDA_
#define _UTILS_SPMV_CUDA_

#include "common.h"

#define LONGROW_THRESHOLD 2048
#define SHORTROW_THRESHOLD 8

__global__
void spmv_longrow_csr_cuda_executor(const int*         d_csrRowPtr,
                          const int*         d_csrColIdx,
                          const VALUE_TYPE*  d_csrVal,
                          const VALUE_TYPE*        d_x,
                          VALUE_TYPE*        d_y,
                          const int longrow,
                          const int*         d_longrow_idx)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    for (int i = 0; i < longrow; i++)
    {
        const int rowid = d_longrow_idx[i];
        const int start = d_csrRowPtr[rowid];
        const int stop  = d_csrRowPtr[rowid+1];
        const int len = stop - start;
        VALUE_TYPE sum = 0;

        if (global_id < len)
        {
            sum = d_x[d_csrColIdx[start+global_id]] * d_csrVal[start+global_id];
        }

        sum = sum_32_shfl(sum);
        if (lane_id == 0 && sum != 0) 
            atomicAdd(&d_y[rowid], sum);
    }
}

__global__
void spmv_threadsca_csr_cuda_executor(const int*         d_csrRowPtr,
                          const int*         d_csrColIdx,
                          const VALUE_TYPE*  d_csrVal,
                          const int          m,
                          const VALUE_TYPE*        d_x,
                          VALUE_TYPE*        d_y)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < m)
    {
        const int rowid = global_id;
        const int start = d_csrRowPtr[rowid];
        const int stop  = d_csrRowPtr[rowid+1];
        VALUE_TYPE sum = 0;
        if (stop - start <= LONGROW_THRESHOLD)
        {
            for (int j = start; j < stop; j++)
                sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        d_y[rowid] = sum;
    }
}

__global__
void spmv_threadsca_dcsr_cuda_executor(const int*         d_csrRowPtr,
                          const int*         d_csrColIdx,
                          const VALUE_TYPE*  d_csrVal,
                          const int          m,
                          const VALUE_TYPE*        d_x,
                          VALUE_TYPE*        d_y,
                          const int          *d_row_perm)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < m)
    {
        const int rowid = global_id;
        const int start = d_csrRowPtr[rowid];
        const int stop  = d_csrRowPtr[rowid+1];
        VALUE_TYPE sum = 0;
        if (stop - start <= LONGROW_THRESHOLD)
        {
            for (int j = start; j < stop; j++)
                sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        //d_y[rowid] = sum;
        d_y[d_row_perm[rowid]] = sum;
    }
}

__global__
void spmv_warpvec_csr_cuda_executor(const int*         d_csrRowPtr,
                          const int*         d_csrColIdx,
                          const VALUE_TYPE*  d_csrVal,
                          const int          m,
                          const VALUE_TYPE*        d_x,
                          VALUE_TYPE*        d_y)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int rowid = global_id / WARP_SIZE;
    if (rowid >= m) return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = d_csrRowPtr[rowid];
    const int stop  = d_csrRowPtr[rowid+1];
    if (start == stop) 
    {
        if (!lane_id) d_y[rowid] = 0;
        return;
    }
   
    VALUE_TYPE sum = 0;
    if (stop - start <= LONGROW_THRESHOLD)
    {
        for (int j = start + lane_id; j < stop; j += WARP_SIZE)
        {
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        sum = sum_32_shfl(sum);
    }

    //finish
    if (!lane_id) d_y[rowid] = sum;
}

__global__
void spmv_warpvec_dcsr_cuda_executor(const int*         d_csrRowPtr,
                          const int*         d_csrColIdx,
                          const VALUE_TYPE*  d_csrVal,
                          const int          m,
                          const VALUE_TYPE*        d_x,
                          VALUE_TYPE*        d_y,
                          const int          *d_row_perm)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int rowid = global_id / WARP_SIZE;
    if (rowid >= m) return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = d_csrRowPtr[rowid];
    const int stop  = d_csrRowPtr[rowid+1];
    if (start == stop) 
    {
        if (!lane_id) d_y[rowid] = 0;
        return;
    }
   
    VALUE_TYPE sum = 0;
    if (stop - start <= LONGROW_THRESHOLD)
    {
        for (int j = start + lane_id; j < stop; j += WARP_SIZE)
        {
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        sum = sum_32_shfl(sum);
    }

    //finish
    if (!lane_id) d_y[d_row_perm[rowid]] = sum; //d_y[rowid] = sum;
}

int spmv_warpvec_csr_cuda(const int           *csrRowPtr,
                          const int           *csrColIdx,
                          const VALUE_TYPE    *csrVal,
                          const int            m,
                          const int            n,
                          const int            nnz,
                          const int            rhs,
                          const int            opt,
                          const VALUE_TYPE    *x,
                                  VALUE_TYPE    *y,
                                  int           *nnzsum,
                                  double        *timesum,
                                  double        *preprocess)
{
    //printf("SpMV: nnz/row = %4.2f\n", (double)nnz/(double)m);
    // compute a y_ref
    memset(y, 0, m * sizeof(VALUE_TYPE));
    if (nnz == 0)
    {
        return 0;
    }
    struct timeval t1, t2;

    VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    for (int i = 0; i < m; i++)
    {
        VALUE_TYPE sum = 0;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++)
            sum += x[csrColIdx[j]] * csrVal[j];
        y_ref[i] = sum;
    }

    // remove empty rows
    int *csrRowPtr_new = (int *)malloc((m+1) * sizeof(int));
    int *row_perm = (int *)malloc(m * sizeof(int));
    csrRowPtr_new[0] = 0;
    int *longrow_idx = (int *)malloc(m * sizeof(int));

    int i_new = 1;
    int lenmax = csrRowPtr[1] - csrRowPtr[0];
    int longrow = 0;
    gettimeofday(&t1, NULL);
    for (int i = 1; i <= m; i++)
    {
        int len = csrRowPtr[i] - csrRowPtr[i-1];
        lenmax = len > lenmax ? len : lenmax;
        //if (len > 5) printf("row[%i] = %i\n", i, len);
        
        if (csrRowPtr[i] != csrRowPtr[i-1])
        {
            csrRowPtr_new[i_new] = csrRowPtr[i];
            row_perm[i_new-1] = i-1;
            if (len > LONGROW_THRESHOLD)
            {
                longrow_idx[longrow] = i_new-1;
                longrow++;
            }
            i_new++;
        }
    }
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    
    int m_new = i_new - 1;
    /*if (nnz != 0)
    {
        printf("SpMV: nnz/row = %4.2f, #empty row %4.2f %%, m_new = %i, nnz = %i %i, lenmax = %i, #longrow = %i\n", 
           (double)nnz/(double)m_new, 100*(double)(m-m_new)/(double)m, m_new, csrRowPtr_new[m_new], csrRowPtr[m], lenmax, longrow);
        for (int i = 0; i < longrow; i++)
            printf("longrow_idx[%i] idx = %i, length = %i\n", 
                   i, longrow_idx[i], csrRowPtr[longrow_idx[i]+1] - csrRowPtr[longrow_idx[i]]);
    }*/

    // transfer host mem to device mem
    int *d_csrRowPtr;
    int *d_csrRowPtr_new;
    int *d_csrColIdx;
    VALUE_TYPE *d_csrVal;
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;
    VALUE_TYPE *d_y_new;
    int *d_row_perm;

    // Matrix L
    cudaMalloc((void **)&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMalloc((void **)&d_csrRowPtr_new, (m_new+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdx,  nnz  * sizeof(int));
    cudaMalloc((void **)&d_csrVal,     nnz  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr_new, csrRowPtr_new, (m_new+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx,  nnz  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,     nnz  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    cudaMemcpy(d_x, x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    
    // Vector y
    cudaMalloc((void **)&d_y, m * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_y, 0, m * rhs * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_y_new, m_new * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_y_new, 0, m_new * rhs * sizeof(VALUE_TYPE));

    // Vector perm
    cudaMalloc((void **)&d_row_perm, m * sizeof(int));
    cudaMemset(d_row_perm, 0, m * sizeof(int));
    cudaMemcpy(d_row_perm, row_perm,  m  * sizeof(int),   cudaMemcpyHostToDevice);

    int *d_longrow_idx;
    cudaMalloc((void **)&d_longrow_idx,  longrow * sizeof(int));
    cudaMemcpy(d_longrow_idx, longrow_idx,  longrow * sizeof(int),   cudaMemcpyHostToDevice);

    //  - cuda reblocking SpMVstart!
    //printf(" - cuda reblocking SpMV start!\n");

    double time_cuda_spmv = 0;
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        if (rhs == 1)
        {
            if (nnz != 0)
            {
                if ((nnz / m_new) < SHORTROW_THRESHOLD)
                {
                    //printf("SPMV SCA-CSR: nnzr = %i\n", nnz / m_new);
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil ((double)m_new / (double)num_threads);
                    spmv_threadsca_csr_cuda_executor<<< num_blocks, num_threads >>>
                                         (d_csrRowPtr_new, d_csrColIdx, d_csrVal,
                                          m_new, d_x, d_y_new);
                }
                else
                {
                    //printf("SPMV VEC-CSR: nnzr = %i\n", nnz / m_new);
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil ((double)m_new / (double)(num_threads/WARP_SIZE));
                    spmv_warpvec_csr_cuda_executor<<< num_blocks, num_threads >>>
                                         (d_csrRowPtr_new, d_csrColIdx, d_csrVal,
                                          m_new, d_x, d_y_new);
                }

                //process long rows
                if (longrow != 0)
                {
                    //printf("SPMV LONG ROW: #longrow = %i\n", longrow);
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil ((double)lenmax / (double)num_threads);
                    spmv_longrow_csr_cuda_executor<<< num_blocks, num_threads >>>
                                            (d_csrRowPtr_new, d_csrColIdx, d_csrVal,
                                            d_x, d_y_new, longrow, d_longrow_idx);
                }
            }
        }
        else
        {
            //num_threads = 4 * WARP_SIZE;
            //num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            //sptrsm_syncfree_reorder_cuda_executor_update<<< num_blocks, num_threads >>>
            //                             (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
            //                              d_graphInDegree, d_left_sum,
            //                              m, substitution, rhs, opt,
            //                              d_b, d_x, d_while_profiler, d_id_extractor);
        }


    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    time_cuda_spmv += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    time_cuda_spmv /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnz;

    //printf("cuda reblocking SpMV used %4.2f ms, throughput is %4.2f gflops, ",
    //       time_cuda_spmv, flop/(1e6*time_cuda_spmv));
    //*gflops = flop/(1e6*time_cuda_spmv);

    VALUE_TYPE *y_new = (VALUE_TYPE *)malloc(m_new * rhs * sizeof(VALUE_TYPE));
   
    cudaMemcpy(y_new, d_y_new, m_new * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    memset(y, 0, m * rhs * sizeof(VALUE_TYPE));
    
    gettimeofday(&t1, NULL);
    for(int i = 0; i < m_new; i++)
        y[row_perm[i]] = y_new[i];
    gettimeofday(&t2, NULL);
    *preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // validate y
    /*double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(y_ref[i]);
        res += abs(y[i] - y_ref[i]);
        //if (y_ref[i] != y[i]) printf ("[%i, %i] y_ref = %f, y = %f\n", i/rhs, i%rhs, y_ref[i], y[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("check passed! |y-yref|/|yref| = %8.2e\n", res);
    else
        printf("check _NOT_ passed! |y-yref|/|yref| = %8.2e\n", res);*/









/*

    if (nnz != 0)
    {
        int BENCH_REPEAT_KERNEL = 100;

        // test kernel
        gettimeofday(&t1, NULL);
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil ((double)m_new / (double)num_threads);
            spmv_threadsca_dcsr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtr_new, d_csrColIdx, d_csrVal,
                                    m_new, d_x, d_y, d_row_perm);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        double time_cuda_spmv_dcsr_sca = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_cuda_spmv_dcsr_sca /= BENCH_REPEAT_KERNEL;

        gettimeofday(&t1, NULL);
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil ((double)m_new / (double)(num_threads/WARP_SIZE));
            spmv_warpvec_dcsr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtr_new, d_csrColIdx, d_csrVal,
                                    m_new, d_x, d_y, d_row_perm);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        double time_cuda_spmv_dcsr_vec = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_cuda_spmv_dcsr_vec /= BENCH_REPEAT_KERNEL;

        // test kernel
        gettimeofday(&t1, NULL);
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil ((double)m / (double)num_threads);
            spmv_threadsca_csr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtr, d_csrColIdx, d_csrVal,
                                    m, d_x, d_y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        double time_cuda_spmv_csr_sca = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_cuda_spmv_csr_sca /= BENCH_REPEAT_KERNEL;

        gettimeofday(&t1, NULL);
        for (int i = 0; i < BENCH_REPEAT_KERNEL; i++)
        {
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            spmv_warpvec_csr_cuda_executor<<< num_blocks, num_threads >>>
                                    (d_csrRowPtr, d_csrColIdx, d_csrVal,
                                    m, d_x, d_y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        double time_cuda_spmv_csr_vec = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_cuda_spmv_csr_vec /= BENCH_REPEAT_KERNEL;


        int nnzr = nnz / m_new;
        double empty_ratio = 100*(double)(m-m_new)/(double)m;

        double time_spmv[4];
        time_spmv[0] = time_cuda_spmv_csr_sca;
        time_spmv[1] = time_cuda_spmv_csr_vec;
        time_spmv[2] = time_cuda_spmv_dcsr_sca;
        time_spmv[3] = time_cuda_spmv_dcsr_vec;

        int best_idx[4] = {0, 1, 2, 3};
        quicksort_keyval<double, int>(time_spmv, best_idx, 0, 4-1);
        int selected = best_idx[0];
        
        //int selected = time_cuda_spmv_dcsr_sca < time_cuda_spmv_dcsr_vec ? 0 : 1;
        //double error = abs(time_cuda_spmv_dcsr_vec - time_cuda_spmv_dcsr_sca);
        //double rate = error / (time_cuda_spmv_dcsr_vec > time_cuda_spmv_dcsr_sca ? time_cuda_spmv_dcsr_sca : time_cuda_spmv_dcsr_vec);
        //selected = rate < 0.03 ? 2 : selected;

        FILE *fout;
        if (sizeof(VALUE_TYPE) == 8)
            fout = fopen("results-spmv-dp-kernel-nnzr.csv", "a");
        else
            fout = fopen("results-spmv-sp-kernel-nnzr.csv", "a");
    
        if (fout == NULL) printf("Writing results fails.\n");
        fprintf(fout, "%i,%i,%i,%f,%f,%f,%f,%i,%f,%i\n", 
                m_new, m, nnz, 
                time_cuda_spmv_csr_sca, time_cuda_spmv_csr_vec, time_cuda_spmv_dcsr_sca, time_cuda_spmv_dcsr_vec, 
                nnzr, empty_ratio, selected);
        fclose(fout);
    }

*/






    cudaFree(d_csrRowPtr);
    cudaFree(d_csrRowPtr_new);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_new);
    cudaFree(d_longrow_idx);
    cudaFree(d_row_perm);

    free(y_ref);
    free(csrRowPtr_new);
    free(row_perm);
    free(y_new);
    free(longrow_idx);

    *nnzsum += nnz;
    *timesum += time_cuda_spmv;

    return 0;
}


#endif
