#include <stdio.h>
#include <iostream>
using namespace std;

#include "helper.h"


void testNull(Matrix2d M) {
        FullPivLU<Matrix2d> lu(M);

        NullResults *R = (NullResults *) malloc(sizeof(NullResults));
        SolveNull(M, R);

        cout << "kernel" << endl;
        cout << lu.kernel() << endl;
        cout << R->kernel << endl;
        cout << "\n" << endl;
}

void testEigen(Matrix2d M) {
        EigenSolver<Matrix2d> eu(M);

        EigenResults *R = (EigenResults *) malloc(sizeof(EigenResults));
        SolveEigen(M, R);

        cout << "eigenvalues" << endl;
        cout << eu.eigenvalues() << endl;
        cout << R->eigenvalues << endl;
        cout << "\n" << endl;

        cout << "eigenvectors" << endl;
        cout << eu.eigenvectors() << endl;
        cout << R->eigenvectors << endl;
        cout << "\n" << endl;
}

void testSVD(Matrix2d M) {
        JacobiSVD<Matrix2d> svd(M, ComputeFullU | ComputeFullV);
        Matrix2d V = svd.matrixV();
        Matrix2d U = svd.matrixU();
        Matrix2d S = Matrix2d(svd.singularValues().asDiagonal());

        SVDResults *R = (SVDResults *) malloc(sizeof(SVDResults));
        SolveJacobiSVD(M, R);

        cout << "V" << endl;
        cout << V << endl;
        cout << R->V << endl;
        cout << "\n" << endl;

        cout << "U" << endl;
        cout << U << endl;
        cout << R->U << endl;
        cout << "\n" << endl;

        cout << "S" << endl;
        cout << S << endl;
        cout << R->singularValues << endl;
        cout << "\n" << endl;

        cout << "M" << endl;
        cout << M << endl;
        cout << U * S * V.transpose() << endl;
        cout << R->U * R->singularValues * R->V.transpose() << endl;
        cout << "\n" << endl;
}


int main() {

        Matrix2d M = Matrix2d();
        M <<    20.1, 9.0, 
                1.89, 0.19;

        testNull(M);
        testEigen(M);
        testSVD(M); 

        return 0;
}
