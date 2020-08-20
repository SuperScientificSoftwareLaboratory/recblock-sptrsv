#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "tranpose.h"
#include "findlevel.h"

#include "sptrsv_syncfree_serialref.h"
#include "sptrsv_cusparse.h"
#include "sptrsv_syncfree_cuda.h"
#include "sptrsv_syncfree_reorder_cuda.h"
#include "sptrsv_recblocking_lv_cuda.h"
#include "sptrsv_reorder_recblocking_lv_cuda.h"
#include "sptrsv_reorder_rowblocking_lv_cuda.h"
#include "sptrsv_reorder_colblocking_lv_cuda.h"

int main(int argc, char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    int nnzTR;
    int *cscRowIdxTR;
    int *cscColPtrTR;
    VALUE_TYPE *cscValTR;

    int *csrRowPtr_tmp;
    int *csrColIdx_tmp;
    VALUE_TYPE *csrVal_tmp;

    int device_id = 0;
    int rhs = 0;
    int lv = 0;
    int substitution = SUBSTITUTION_FORWARD;

    // "Usage: ``./sptrsv -d 0 -rhs 1 -lv 3 -forward -mtx A.mtx'' for LX=B on device 0"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);

    // load the number of right-hand-side
    char *rhsstr;
    if(argc > argi)
    {
        rhsstr = argv[argi];
        argi++;
    }

    if (strcmp(rhsstr, "-rhs") != 0) return 0;

    if(argc > argi)
    {
        rhs = atoi(argv[argi]);
        argi++;
    }
    printf("rhs = %i\n", rhs);

    // load the number of recursive levels
    char *lvstr;
    if(argc > argi)
    {
        lvstr = argv[argi];
        argi++;
    }

    if (strcmp(lvstr, "-lv") != 0) return 0;

    if(argc > argi)
    {
        lv = atoi(argv[argi]);
        argi++;
    }
    printf("lv = %i\n", lv);

    // load substitution, forward or backward
    char *substitutionstr;
    if(argc > argi)
    {
        substitutionstr = argv[argi];
        argi++;
    }

    if (strcmp(substitutionstr, "-forward") == 0)
        substitution = SUBSTITUTION_FORWARD;
    else if (strcmp(substitutionstr, "-backward") == 0)
        substitution = SUBSTITUTION_BACKWARD;
    printf("substitutionstr = %s\n", substitutionstr);
    printf("substitution = %i\n", substitution);

    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    printf("matstr = %s\n", matstr);

    // load matrix data from file
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("-------------- %s --------------\n", filename);

    srand(time(NULL));
    if (strcmp(matstr, "-mtx") == 0)
    {
        // load mtx data to the csr format
        mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
        csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
        csrColIdxA = (int *)malloc(nnzA * sizeof(int));
        csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
        mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
        printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
        
        if (m!=n)
        {
            printf("we need square matrix. Exit!\n");
            return 0;
        }
        //if (nnzA < 2000000 || nnzA > 200000000 || m < 100000)
	//if (nnzA >= 2000000 || m >= 100000)
        //{
        //    printf("This code only computes matrices with 2M <= nnz <= 200M and m >= 100k. Exit!\n");
        //    return 0;
        //}

        // extract L or U with a unit diagonal of A
        csrRowPtr_tmp = (int *)malloc((m+1) * sizeof(int));
        csrColIdx_tmp = (int *)malloc((m+nnzA) * sizeof(int));
        csrVal_tmp    = (VALUE_TYPE *)malloc((m+nnzA) * sizeof(VALUE_TYPE));

        int nnz_pointer = 0;
        csrRowPtr_tmp[0] = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {   
                if (substitution == SUBSTITUTION_FORWARD)
                {
                    if (csrColIdxA[j] < i)
                    {
                        csrColIdx_tmp[nnz_pointer] = csrColIdxA[j];
                        csrVal_tmp[nnz_pointer] = rand() % 10 + 1; //csrValA[j]; 
                        nnz_pointer++;
                    }
                }
                else if (substitution == SUBSTITUTION_BACKWARD)
                {
                    if (csrColIdxA[j] > i)
                    {
                        csrColIdx_tmp[nnz_pointer] = csrColIdxA[j];
                        csrVal_tmp[nnz_pointer] = rand() % 10 + 1; //csrValA[j]; 
                        nnz_pointer++;
                    }
                }
            }

            // add dia nonzero
            csrColIdx_tmp[nnz_pointer] = i;
            csrVal_tmp[nnz_pointer] = 1.0;
            nnz_pointer++;

            csrRowPtr_tmp[i+1] = nnz_pointer;
        }

        int nnz_tmp = csrRowPtr_tmp[m];
        nnzTR = nnz_tmp;

        if (substitution == SUBSTITUTION_FORWARD)
            printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        else if (substitution == SUBSTITUTION_BACKWARD)
            printf("A's unit-upper triangular U: ( %i, %i ) nnz = %i\n", m, n, nnzTR);

        csrColIdx_tmp = (int *)realloc(csrColIdx_tmp, sizeof(int) * nnzTR);
        csrVal_tmp = (VALUE_TYPE *)realloc(csrVal_tmp, sizeof(VALUE_TYPE) * nnzTR);

        cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));
        cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrTR, 0, (n+1) * sizeof(int));
        cscValTR    = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

        // transpose from csr to csc
        matrix_transposition(m, n, nnzTR,
                             csrRowPtr_tmp, csrColIdx_tmp, csrVal_tmp,
                             cscRowIdxTR, cscColPtrTR, cscValTR);

        // keep each column sorted
        //#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            //quick_sort_key_val_pair<int, VALUE_TYPE>(&cscRowIdxTR[cscColPtrTR[i]],
            //                                  &cscValTR[cscColPtrTR[i]],
            //                                  cscColPtrTR[i+1]-cscColPtrTR[i]);
        }

        // check unit diagonal
        int dia_miss = 0;
        for (int i = 0; i < n; i++)
        {
            bool miss;
            if (substitution == SUBSTITUTION_FORWARD)
                miss = cscRowIdxTR[cscColPtrTR[i]] != i;
            else if (substitution == SUBSTITUTION_BACKWARD)
                cscRowIdxTR[cscColPtrTR[i+1] - 1] != i;

            if (miss) dia_miss++;
        }
        //printf("dia miss = %i\n", dia_miss);
        if (dia_miss != 0) 
        {
            printf("This matrix has incomplete diagonal, #missed dia nnz = %i\n", dia_miss); 
            return;
        }



        free(csrColIdxA);
        free(csrValA);
        free(csrRowPtrA);
    }
    else if (strcmp(matstr, "-csc") == 0)
    {
        FILE *f;
        int returnvalue;

        if ((f = fopen(filename, "r")) == NULL)
            return -1;

        returnvalue = fscanf(f, "%d", &m);
        returnvalue = fscanf(f, "%d", &n);
        returnvalue = fscanf(f, "%d", &nnzTR);

        cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrTR, 0, (n+1) * sizeof(int));
        cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));
        cscValTR    = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

        // read row idx
        for (int i = 0; i < n+1; i++)
        {
            returnvalue = fscanf(f, "%d", &cscColPtrTR[i]);
            cscColPtrTR[i]--; // from 1-based to 0-based
        }

        // read col idx
        for (int i = 0; i < nnzTR; i++)
        {
            returnvalue = fscanf(f, "%d", &cscRowIdxTR[i]);
            cscRowIdxTR[i]--; // from 1-based to 0-based
        }

        // read val
        for (int i = 0; i < nnzTR; i++)
        {
            cscValTR[i] = rand() % 10 + 1;
            //returnvalue = fscanf(f, "%lg", &cscValTR[i]);
        }

        if (f != stdin)
            fclose(f);

        // keep each column sorted
        for (int i = 0; i < n; i++)
        {
            //quick_sort_key_val_pair<int, int>(&cscRowIdxTR[cscColPtrTR[i]],
            //                                  &cscRowIdxTR[cscColPtrTR[i]],
            //                                  cscColPtrTR[i+1]-cscColPtrTR[i]);
            quicksort_keyval<int, VALUE_TYPE>(&cscRowIdxTR[cscColPtrTR[i]], &cscValTR[cscColPtrTR[i]], 
                                              0, cscColPtrTR[i+1]-cscColPtrTR[i]);
        }

        if (substitution == SUBSTITUTION_FORWARD)
            printf("Input csc unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        else if (substitution == SUBSTITUTION_BACKWARD)
            printf("Input csc unit-upper triangular U: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
       
        // check unit diagonal
        int dia_miss = 0;
        for (int i = 0; i < n; i++)
        {
            bool miss;
            if (substitution == SUBSTITUTION_FORWARD)
                miss = cscRowIdxTR[cscColPtrTR[i]] != i;
            else if (substitution == SUBSTITUTION_BACKWARD)
                cscRowIdxTR[cscColPtrTR[i+1] - 1] != i;

            if (miss) dia_miss++;
        }
        //printf("dia miss = %i\n", dia_miss);
        if (dia_miss != 0) 
        {
            printf("This matrix has incomplete diagonal, #missed dia nnz = %i\n", dia_miss); 
            return;
        }
    }

    // find level sets
    int nlevel = 0;
    int parallelism_min = 0;
    int parallelism_avg = 0;
    int parallelism_max = 0;
    findlevel_csc(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR, &nlevel,
                  &parallelism_min, &parallelism_avg, &parallelism_max);
    double fparallelism = (double)m/(double)nlevel;
    printf("This matrix/graph has %i levels, its parallelism is %4.2f (min: %i ; avg: %i ; max: %i )\n", 
           nlevel, fparallelism, parallelism_min, parallelism_avg, parallelism_max);

    // x and b are all row-major
    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    for ( int i = 0; i < n; i++)
        for (int j = 0; j < rhs; j++)
            x_ref[i * rhs + j] = rand() % 10 + 1; //j + 1;

    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);

    for (int i = 0; i < m * rhs; i++)
        b[i] = 0;

    for (int i = 0; i < n * rhs; i++)
        x[i] = 0;

    // run csc spmv to generate b
    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtrTR[i]; j < cscColPtrTR[i+1]; j++)
        {
            int rowid = cscRowIdxTR[j]; //printf("rowid = %i\n", rowid);
            for (int k = 0; k < rhs; k++)
            {
                b[rowid * rhs + k] += cscValTR[j] * x_ref[i * rhs + k];
            }
        }
    }

    // run serial syncfree SpTRSV as a reference
    printf("---------------------------------------------------------------------------------------------\n");
    //sptrsv_syncfree_serialref(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
    //                          substitution, rhs, x, b, x_ref);

    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);

    // run cuSPARSE
    printf("---------------------------------------------------------------------------------------------\n");
    double gflops_csrsv = 0;
    double preprocessing_csrsv = 0;
    double runtime_csrsv = 0;
    double gflops_csrsv2 = 0;
    double preprocessing_csrsv2 = 0;
    double runtime_csrsv2 = 0;
    sptrsv_cusparse(csrRowPtr_tmp, csrColIdx_tmp, csrVal_tmp, m, n, nnzTR, x, b, x_ref, 
                    &preprocessing_csrsv, &runtime_csrsv, &gflops_csrsv, &preprocessing_csrsv2, &runtime_csrsv2, &gflops_csrsv2);
    // write results to text (scv) file
    FILE *fout = fopen("results-cusparse.csv", "a");
    if (fout == NULL) printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,%f,%f,%f,%f,%f,%f\n", 
            sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", 
            filename, m, nnzTR, preprocessing_csrsv, runtime_csrsv, gflops_csrsv, preprocessing_csrsv2, runtime_csrsv2, gflops_csrsv2);
    fclose(fout);
    printf("---------------------------------------------------------------------------------------------\n");

    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowPtr_tmp);

    // run cuda syncfree SpTRSV or SpTRSM
    printf("---------------------------------------------------------------------------------------------\n");
    double gflops_syncfree = 0;
    double preprocessing_syncfree = 0;
    double runtime_syncfree = 0;
    memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
    sptrsv_syncfree_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                         substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, &preprocessing_syncfree, &runtime_syncfree, &gflops_syncfree);
    fout = fopen("results-syncfree.csv", "a");
    if (fout == NULL) printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,%i,%f,%i,%i,%i,%f,%f,%f\n", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", 
            filename, m, nnzTR, nlevel, fparallelism, parallelism_min, parallelism_avg, parallelism_max, 
            preprocessing_syncfree, runtime_syncfree, gflops_syncfree);
    fclose(fout);
    printf("---------------------------------------------------------------------------------------------\n");

    //printf("---------------------------------------------------------------------------------------------\n");
    //gflops_autotuned = 0;
    //memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
    //sptrsv_syncfree_reorder_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
    //                     substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, &gflops_autotuned);
    //printf("---------------------------------------------------------------------------------------------\n");

    printf("---------------------------------------------------------------------------------------------\n");
    double flop = 2*(double)rhs*(double)nnzTR;
    double gb = (m+1)*sizeof(int) + nnzTR*(sizeof(int)+sizeof(VALUE_TYPE)) + 2*m*rhs*sizeof(VALUE_TYPE);
    int nnz_sptrsv = 0;
    int nnz_spmv = 0;
    double timesum_sptrsv = 0;
    double timesum_spmv = 0;
    double preprocessing_blocking = 0;
    //memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
    //sptrsv_reblocking_lv_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
    //                     substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, lv, 
    //                     &nnz_sptrsv, &nnz_spmv, &timesum_sptrsv, &timesum_spmv);
    //printf("CUDA reblocking SpTRSV (#level = %i) solve used \n(SpTRSV-tri nnz %10i, time %4.2f ms) + \n(SpMV-sqr   nnz %10i, time %4.2f ms), throughput is %4.2f gflops or %4.2f GB/s\n",
    //       lv, nnz_sptrsv, nnz_spmv, timesum_sptrsv, timesum_spmv, 
    //       flop/(1e6*(timesum_sptrsv + timesum_spmv)), gb/(1e6*(timesum_sptrsv + timesum_spmv)));
    //printf("---------------------------------------------------------------------------------------------\n");

    printf("---------------------------------------------------------------------------------------------\n");
    FILE *foutnnz = fopen("results-recblocking-nnz.csv", "a");
    if (foutnnz == NULL) printf("Writing results fails.\n");
    fprintf(foutnnz, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    FILE *fouttime = fopen("results-recblocking-time.csv", "a");
    if (fouttime == NULL) printf("Writing results fails.\n");
    fprintf(fouttime, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    fout = fopen("results-recblocking.csv", "a");
    if (fout == NULL) printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    if (lv == -1)
    {
        int li = 1;
        for (li = 1; li <= 100; li++)
        {
            if (m / pow(2, (li+1)) < (device_id == 0 ? 92160 : 58880)) // 92160 (4608x20) is titan rtx, 58880 (2944x20) is rtx 2080
                break;
        }
        lv = li;
        //printf("lv = %i\n", lv);
    }

    for (int li = lv; li <= lv; li++)
    {
        //if (m / pow(2, li) < 8192)
        //    break;

        nnz_sptrsv = 0;
        nnz_spmv = 0;
        timesum_sptrsv = 0;
        timesum_spmv = 0;
        preprocessing_blocking = 0;
        memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
        sptrsv_reorder_recblocking_lv_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                            substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, li, 
                            &nnz_sptrsv, &nnz_spmv, &timesum_sptrsv, &timesum_spmv, &preprocessing_blocking);
        printf("CUDA reorder-reblocking SpTRSV (#level = %i) solve used \n preprocessing time = %4.2f ms, sptrsv time %4.2f ms\n(SpTRSV-tri nnz %10i, time %4.2f ms) + \n(SpMV-sqr   nnz %10i, time %4.2f ms), throughput is %4.2f gflops or %4.2f GB/s\n",
            li, preprocessing_blocking, timesum_sptrsv + timesum_spmv, nnz_sptrsv, timesum_sptrsv, nnz_spmv, timesum_spmv, 
            flop/(1e6*(timesum_sptrsv + timesum_spmv)), gb/(1e6*(timesum_sptrsv + timesum_spmv)));

        fprintf(fout, "%i,%f,", li, flop/(1e6*(timesum_sptrsv + timesum_spmv)));
        fprintf(foutnnz, "%i,%i,", nnz_sptrsv, nnz_spmv);
        fprintf(fouttime, "%f,%f,", preprocessing_blocking, timesum_sptrsv + timesum_spmv);
    }
    fprintf(fout, "\n");
    fclose(fout);
    fprintf(foutnnz, "\n");
    fclose(foutnnz);
    fprintf(fouttime, "\n");
    fclose(fouttime);
    printf("---------------------------------------------------------------------------------------------\n");

return 0;

    printf("---------------------------------------------------------------------------------------------\n");
    /*foutnnz = fopen("results-rowblocking-nnz.csv", "a");
    if (foutnnz == NULL) printf("Writing results fails.\n");
    fprintf(foutnnz, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    fouttime = fopen("results-rowblocking-time.csv", "a");
    if (fouttime == NULL) printf("Writing results fails.\n");
    fprintf(fouttime, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    fout = fopen("results-rowblocking.csv", "a");
    if (fout == NULL) printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);*/
    for (int li = 2; li <= lv; li++)
    {
        if (m / li < 8192)
            break;
        nnz_sptrsv = 0;
        nnz_spmv = 0;
        timesum_sptrsv = 0;
        timesum_spmv = 0;
        preprocessing_blocking = 0;
        memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
        sptrsv_reorder_rowblocking_lv_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                            substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, li, 
                            &nnz_sptrsv, &nnz_spmv, &timesum_sptrsv, &timesum_spmv, &preprocessing_blocking);
        printf("CUDA reorder-rowblocking SpTRSV (#level = %i) solve used \n preprocessing time = %4.2f ms, sptrsv time %4.2f ms \n(SpTRSV-tri nnz %10i, time %4.2f ms) + \n(SpMV-sqr   nnz %10i, time %4.2f ms), throughput is %4.2f gflops or %4.2f GB/s\n",
            li, preprocessing_blocking, timesum_sptrsv + timesum_spmv, nnz_sptrsv, timesum_sptrsv, nnz_spmv, timesum_spmv, 
            flop/(1e6*(timesum_sptrsv + timesum_spmv)), gb/(1e6*(timesum_sptrsv + timesum_spmv)));

        /*fprintf(fout, "%f,", flop/(1e6*(timesum_sptrsv + timesum_spmv)));
        fprintf(foutnnz, "%i,%i,", nnz_sptrsv, nnz_spmv);
        fprintf(fouttime, "%f,%f,", preprocessing_blocking, timesum_sptrsv + timesum_spmv);*/
    }
    /*fprintf(fout, "\n");
    fclose(fout);
    fprintf(foutnnz, "\n");
    fclose(foutnnz);
    fprintf(fouttime, "\n");
    fclose(fouttime);*/
    printf("---------------------------------------------------------------------------------------------\n");

    printf("---------------------------------------------------------------------------------------------\n");
    /*foutnnz = fopen("results-colblocking-nnz.csv", "a");
    if (foutnnz == NULL) printf("Writing results fails.\n");
    fprintf(foutnnz, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    fouttime = fopen("results-colblocking-time.csv", "a");
    if (fouttime == NULL) printf("Writing results fails.\n");
    fprintf(fouttime, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);

    fout = fopen("results-colblocking.csv", "a");
    if (fout == NULL) printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,", sizeof(VALUE_TYPE) == 8 ? "DOUBLE" : "FLOAT", filename, m, nnzTR);*/
    for (int li = 2; li <= lv; li++)
    {
        if (m / li < 8192)
            break;

        nnz_sptrsv = 0;
        nnz_spmv = 0;
        timesum_sptrsv = 0;
        timesum_spmv = 0;
        preprocessing_blocking = 0;
        memset(x, 0, sizeof(VALUE_TYPE) * n * rhs);
        sptrsv_reorder_colblocking_lv_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                            substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, li, 
                            &nnz_sptrsv, &nnz_spmv, &timesum_sptrsv, &timesum_spmv, &preprocessing_blocking);
        printf("CUDA reorder-colblocking SpTRSV (#level = %i) solve used \n preprocessing time = %4.2f ms, sptrsv time %4.2f ms \n(SpTRSV-tri nnz %10i, time %4.2f ms) + \n(SpMV-sqr   nnz %10i, time %4.2f ms), throughput is %4.2f gflops or %4.2f GB/s\n",
            li, preprocessing_blocking, timesum_sptrsv + timesum_spmv, nnz_sptrsv, timesum_sptrsv, nnz_spmv, timesum_spmv, 
            flop/(1e6*(timesum_sptrsv + timesum_spmv)), gb/(1e6*(timesum_sptrsv + timesum_spmv)));

        /*fprintf(fout, "%f,", flop/(1e6*(timesum_sptrsv + timesum_spmv)));
        fprintf(foutnnz, "%i,%i,", nnz_sptrsv, nnz_spmv);
        fprintf(fouttime, "%f,%f,", preprocessing_blocking, timesum_sptrsv + timesum_spmv);*/
    }
    /*fprintf(fout, "\n");
    fclose(fout);
    fprintf(foutnnz, "\n");
    fclose(foutnnz);
    fprintf(fouttime, "\n");
    fclose(fouttime);*/
    printf("---------------------------------------------------------------------------------------------\n");

    // done!
    free(cscRowIdxTR);
    free(cscColPtrTR);
    free(cscValTR);

    free(x);
    free(x_ref);
    free(b);

    return 0;
}
