# recblock-sptrsv
Source code of paper ``Efficient Block Algorithms for Parallel Sparse Triangular Solve'' at ICPP 2020
## paper infromation
Zhengyang Lu, Yuyao Niu, and Weifeng Liu. 2020. Efficient Block Algorithms for Parallel Sparse Triangular Solve. In 49th International Conference on Parallel Processing - ICPP (ICPP '20). Association for Computing Machinery, New York, NY, USA, Article 63, 1–11. DOI:https://doi.org/10.1145/3404397.3404413
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
  
