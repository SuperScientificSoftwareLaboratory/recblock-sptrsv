# recblock-sptrsv
**recblock-sptrsv** implements recursive block algorithms for parallel SpTRSV on modern GPUs, and propose an adaptive approach that can automatically select the best kernels according to input sparsity structures.
## Paper information
Zhengyang Lu, Yuyao Niu, and Weifeng Liu. 2020. Efficient Block Algorithms for Parallel Sparse Triangular Solve. In 49th International Conference on Parallel Processing - ICPP (ICPP '20). Association for Computing Machinery, New York, NY, USA, Article 63, 1–11. DOI:https://doi.org/10.1145/3404397.3404413
# Introduction
The sparse triangular solve (SpTRSV) operation solves a linear system of the form 𝐿𝑥 = 𝑏 (or 𝑈 𝑥 = 𝑏), where 𝐿 (or 𝑈 ) is a sparse lower (or upper) triangular matrix, 𝑏 is a dense right-hand side vector, and 𝑥 is the dense resulting vector to solve.
recblock-sptrsv implement recursive blocks (each including two triangular sub-matrices and a square or near square sub-matrix). And we further improve its performance by using a
new data format. In addition, we propose an adaptive method that automatically selects the best SpTRSV and SpMV kernels for the divided triangular sub-matrices and square sub-matrices, respectively, depending on their sparsity structures
# Installation
NVIDIA GPU with compute capability at least 3.5 (NVIDIA Titan X and Titan RTX as tested) * NVIDIA nvcc CUDA compiler and cuSPARSE library, both of which are included with CUDA Toolkit (CUDA v10.2 as tested) The GPU test programs have been tested on Ubuntu 18.04/20.04, and are expected to run correctly under other Linux distributions.
# Execution of recblock-sptrsv
Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection.

# Usage examples:
```
./sptrsv-float -d 1 -rhs 1 -lv -1 -forward -mtx Name.mtx.     
./sptrsv-double -d 1 -rhs 1 -lv -1 -forward -mtx Name.mtx.
```
# Parameters:
## Required:
  * ***matrix***: path to the input matrix file stored in Matrix Market Format.   
  * ***datatype***: sptrsv-float(float) or sptrsv-double(double).   
## Optional:
  * ***-d***: id of the GPU to execute.   
  * ***-rhs***: the number of right-hand-side vectors.   
  * ***-lv***: the level number of recblock-sptrsv algorithm.   
  * ***-forward/-backward***: Lower/Upper triangular matrix.   
  * ***-mtx***: matrix file stored in Matrix Market Format.   
  
