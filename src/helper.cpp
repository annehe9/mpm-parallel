#include "helper.h"

#define EPSILON 0.00000001

/* Operations to replace eigen on CUDA, for 2x2 matrices. */

double determinant(Matrix2d M) {
    return M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
}

// normalized kernel
void SolveNull(Matrix2d M, NullResults* R) {
    double a = M(0, 0);
    double b = M(0, 1);
    double c = M(1, 0);
    double d = M(1, 1);

    R->kernel = Matrix2d::Zero();
    if (a == 0 && b == 0 && c == 0 && d == 0) {
        R->kernel = Matrix2d::Identity(); // full nullity
        R->nullity = 2;
    }
    else if (b == 0 && d == 0) { // one of a or c is non-zero
        R->kernel.col(0) << 0, 1;
        R->nullity = 1;
    }
    else if (a == 0 && c == 0) { // one of b or d is non-zero
        R->kernel.col(0) << 1, 0;
    }
    else if (c == 0 && d == 0) { // a and b are non-zero
        R->kernel.col(0) << -b / a, 1;
        R->kernel.col(0).normalize();
        R->nullity = 1;
    }
    else if (a == 0 && b == 0) {
        R->kernel.col(0) << -d / c, 1;
        R->kernel.col(0).normalize();
        R->nullity = 1;
    }
    else if (a != 0 && b != 0 && c != 0 && d != 0) { // all 4 non-zero
        if (b / a - d / c < EPSILON) {
            R->kernel.col(0) << -b / a, 1;
            R->kernel.col(0).normalize();
            R->nullity = 1;
        }
        else {
            R->nullity = 0;
        }
    }
    else { // 3 are non-zero, or diagonal
        R->nullity = 0;
    }
}

// assume square
// normalized eigenvectors
void SolveEigen(Matrix2d M, EigenResults* R) {
    // solve for eigenvalues: 
    // quadratic formula, det(M - lambda * I) = 0 for 2x2
    double a = M(0, 0);
    double b = M(0, 1);
    double c = M(1, 0);
    double d = M(1, 1);
    double part1 = a + d;
    double part2 = sqrt(a * a + d * d - 2 * a * d + 4 * b * c);
    double e1 = (part1 + part2) / 2;
    double e2 = (part1 - part2) / 2;

    R->eigenvalues = Vector2d::Zero();
    R->rank = 0;
    if (e1 != 0) {
        R->eigenvalues(R->rank) = e1;
        R->rank += 1;
    }
    if (e2 != 0) {
        R->eigenvalues(R->rank) = e2;
        R->rank += 1;
    }

    // solve for corresponding eigenvectors -> eigenspace...
    // kernel of M - lambda * I
    R->eigenvectors = Matrix2d::Zero();
    Matrix2d I = Matrix2d::Identity();
    NullResults* N = (NullResults*)malloc(sizeof(NullResults));
    for (int i = 0; i < R->rank; i++) {
        Matrix2d X = M - R->eigenvalues(i) * I;
        SolveNull(X, N);
        R->eigenvectors.col(i) << N->kernel.col(0); // copy
        // case where eigenspace is rank 2
        if (N->nullity == 2) {
            assert(i == 0);
            R->eigenvectors.col(1) << N->kernel.col(1); // copy
            break; // we are done
        }
    }
}

// https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
// watch out if M is identity
void SolveJacobiSVD(Matrix2d M, SVDResults* R) {
    R->V = Matrix2d::Zero();
    R->U = Matrix2d::Zero();
    R->singularValues = Matrix2d::Zero();

    // handle special identity matrix case
    if (M(0, 0) == 1 && M(0, 1) == 0 && M(1, 0) == 0 && M(1, 1) == 1) {
       R->V = Matrix2d::Identity();
       R->U = Matrix2d::Identity();
       R->singularValues = Matrix2d::Identity();
       return;
    }

    double y1 = (M(1, 0) + M(0, 1));
    double x1 = (M(0, 0) - M(1, 1));
    double y2 = (M(1, 0) - M(0, 1));
    double x2 = (M(0, 0) + M(1, 1));
    double h1 = hypot(y1, x1);
    double h2 = hypot(y2, x2);
    double t1 = x1 / h1;
    double t2 = x2 / h2;

    double cc = sqrt((1 + t1) * (1 + t2));
    double ss = sqrt((1 - t1) * (1 - t2));
    double cs = sqrt((1 + t1) * (1 - t2));
    double sc = sqrt((1 - t1) * (1 + t2));
    double cn = (cc - ss) / 2;
    double sn = (sc + cs) / 2;

    R->V << cn, -sn,
        sn, cn;

    double sig1 = (h1 + h2) / 2;
    double sig2 = (h1 - h2) / 2;
    Vector2d sig = Vector2d();
    sig << sig1, sig2;
    R->singularValues = Matrix2d(sig.asDiagonal());

    Vector2d sig_inv = Vector2d();
    if (h1 != h2) {
        sig_inv << 1 / sig1, 1 / sig2;
    }
    else {
        sig_inv << 1 / sig1, 0;
    }
    R->U = M * R->V * Matrix2d(sig_inv.asDiagonal());
}


