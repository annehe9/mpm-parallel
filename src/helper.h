#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

#include "eigen3/Eigen/Dense"
using namespace Eigen;

struct NullResults {
        Matrix2d kernel;
        int nullity;
};

struct EigenResults {
        Matrix2d eigenvectors;
        Vector2d eigenvalues; // may have 0's
        int rank;
};

struct SVDResults {
        Matrix2d V;
        Matrix2d U;
        Matrix2d singularValues;
};

__device__ double determinant(Matrix2d M);
__device__ void SolveNull(Matrix2d M, NullResults *R);
__device__ void SolveEigen(Matrix2d M, EigenResults *R);
__device__ void SolveJacobiSVD(Matrix2d M, SVDResults *R);
