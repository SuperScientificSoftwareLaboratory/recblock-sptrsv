# recblock-sptrsv
Source code of paper ``Efficient Block Algorithms for Parallel Sparse Triangular Solve'' at ICPP 2020
# Usage examples:
./sptrsv-float -d 1 -rhs 1 -lv -1 -forward -mtx Name.mtx
./sptrsv-double -d 1 -rhs 1 -lv -1 -forward -mtx Name.mtx
# Parameters:
Required:
  matrix: path to the input matrix file stored in Matrix Market Format
  datatype: sptrsv-float or sptrsv-double
Optional:
  -d: id of the GPU to execute
  -rhs: the number of right-hand-side vectors
  -lv: the level number of recblock-sptrsv algorithm
  -forward/-backward: Lower/Upper triangular matrix
  -mtx: matrix file stored in Matrix Market Format
  
