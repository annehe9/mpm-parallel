#include "helper.h"


#define EPSILON (0.00000000001)

bool equal(double a1, double a2, double eps) {
        if (a1 > a2)
                return a1 - a2 <= eps;
        else 
                return a2 - a1 <= eps;
}

bool equal_mat(Matrix2d A, Matrix2d B, double eps) {
        return equal(A(0, 0), B(0, 0), eps) 
                && equal(A(0, 1), B(0, 1), eps) 
                && equal(A(1, 0), B(1, 0), eps) 
                && equal(A(1, 1), B(1, 1), eps);
}

void print_compare(Matrix2d M, Matrix2d A, Matrix2d B) {
       IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
       cout << M.format(HeavyFmt) << endl;
       cout << A.format(HeavyFmt) << endl;
       cout << B.format(HeavyFmt) << endl;
}

/* Operations to replace eigen on CUDA, for 2x2 matrices. */

double determinant(Matrix2d M) {
        return M(0,0) * M(1,1) - M(0,1) * M(1,0);
}

// normalized kernel
void SolveNull(Matrix2d M, NullResults *R) {
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
                R->kernel.col(0) << -b/a, 1;
                R->kernel.col(0).normalize();
                R->nullity = 1;
        }
        else if (a == 0 && b == 0) {
                R->kernel.col(0) << -d/c, 1;
                R->kernel.col(0).normalize();
                R->nullity = 1;
        }
        else if (a != 0 && b != 0 && c != 0 && d != 0) { // all 4 non-zero
                if (b/a - d/c < EPSILON) {
                        R->kernel.col(0) << -b/a, 1;
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
void SolveEigen(Matrix2d M, EigenResults *R) {
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
        NullResults *N = (NullResults *) malloc(sizeof(NullResults));
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

// performs main work
void SVDHelper(SVDResults *R, Matrix2d work, 
        Index p, Index q, Matrix2d *left, Matrix2d *right) {
        // run for svd_precondition_2x2_block_to_be_real
        // if not real, make real
        /*
        using std::sqrt;
        Scalar z;
        Matrix2d rot;
        JacobiRotation<Scalar>rot;
        RealScalar n = sqrt(numext::abs2(work(p,p)) + numext::abs2(work(q,p)));
        if (n == 0) {
                z = abs(work(p,q) / work(p,q));
                work.row(p) *= z;
                R->U.col(p) *= conj(z);

                z = abs(work(q,q) / work(q,q));
                work.row(q) *= z;
                R->U.col(q) *= conj(z);
        }
        else {
                rot.c() = conj(work(p,p) / n);
                rot.s() = work(q,p) / n;
                work.applyOnTheLeft(p, q, rot);
                R->U.applyOnTheRight(p, q, rot.adjoint());
                if (work(p,q) != Scalar(0)) {
                        Scalar z = abs(work(p,q) / work(p,q));
                        work.col(q) *= z;
                        R->V.col(q) *= z;
                }
                if (work(q,q) != Scalar(0)) {
                        z = abs(work(q,q) / work(q,q));
                        work.row(q) *= z;
                        R->U.col(q) *= conj(z);
                }
        }
        */

        // real_2x2_jacobi_svd
        Matrix<RealScalar, 2, 2> m;
        m <<    numext::real(matrix(p,p)), numext::real(matrix(p,q)),
                numext::real(matrix(q,p)), numext::real(matrix(q,q));

        RealScalar t = m(0,0) + m(1,1); 
        RealScalar d = m(1,0) + m(0,1); 

        JacobiRotation<RealScalar> rot1;
        if (t == RealScalar(0)) {
                rot1.c() = RealScalar(0);
                rot.s() = d > RealScalar(0) ? RealScalar(1) : RealScalar(-1);
        }
        else {
                u = d / t;
                rot1.c() = RealScalar(1) / sqrt(RealScalar(1) + numext::abs2(u));
                rot1.s() = rot1.c() * u;
        }

        m.applyOnTheLeft(0, 1, rot1); // self-adjoint

        // https://en.wikipedia.org/wiki/Jacobi_rotation
        //right->makeJacobi(m, 0, 1); // TODO, find J s.t. J^-1 * m * J is diag
        RealScalar beta = (m(1,1) - m(0,0)) / (2 * m(0,1));
        RealScalar t = sgn(beta) / (abs(beta) + sqrt(beta * beta + 1));
        right.c() = 1 / sqrt(t * t + 1);
        right.s() = right.c() * t;
        // TODO check if want transpose of right

        *left = rot1 * right->transpose();
}

// https://github.com/OPM/eigen3/blob/master/unsupported/Eigen/src/SVD/JacobiSVD.h
void SolveJacobiSVD(Matrix2d M, SVDResults *R) {
        R->V = Matrix2d::Identity();
        R->U = Matrix2d::Identity();
        R->singularValues = Matrix2d::Identity();
        Matrix2d work = M; // copy
        
        using std::max;
        Index p = 1; Index q = 0;
        const RealScalar precision = RealScalar(2) * 
                NumTraits<Scalar>::epsilon();
        const RealScalar considerAsZero = RealScalar(2) * 
                std::numeric_limits<RealScalar>::denorm_min();
        RealScalar threshold = (max)(considerAsZero, 
                precision * (max)(abs(work(p,p)), abs(work(q,q))));

        if ((max)(abs(work(p,q)), abs(work(q,p))) <= threshold) return; // early stop
        JacobiRotation<RealScalar> left, right;

        // diagonalize work and get left and right
        SVDHelper(R, work, p, q, &left, &right);
        R->U.applyOnTheRight(p,q,left.transpose()); //R->U = R->U * left.transpose();
        R->V.applyOnTheRight(p,q,right);//R->V = R->V * right;

        // sigs
        for (int i = 0; i < 2; i++) {
                RealScalar a = abs(work(i, i));
                R->singularValues(i, i) = a;
                if (a != RealScalar(0)) R->U.col(i) *= work(i,i) / a; // sign
        }

        /* TODO: sort sigs in descending order, compute number of nonzero sigs */
}

// https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
/*
void SolveJacobiSVD(Matrix2d M, SVDResults *R) {
       R->V = Matrix2d::Zero();
       R->U = Matrix2d::Zero();
       R->singularValues = Matrix2d::Zero();

       if (M.isApprox(Matrix2d::Identity())) {//equal_mat(M, Matrix2d::Identity(), 0)) {
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

       R->V <<  cn, -sn,
                sn, cn;

       double sig1 = (h1 + h2) / 2;
       if (sig1 < 0) sig1 = -sig1; // fix to prevent neg vals, as suggested
       double sig2 = (h1 - h2) / 2;
       if (sig2 < 0) sig2 = -sig2; // fix to prevent neg vals, as suggested
       Vector2d sig = Vector2d();
       sig << sig1, sig2;
       R->singularValues = Matrix2d(sig.asDiagonal());

       Vector2d sig_inv = Vector2d();
       if (h1 != h2) {
               sig_inv << 1 / sig1, 1 / sig2;
       }
       else { // determinant is 0
               sig_inv << 1 / sig1, 0;
       }
       R->U = M * R->V * Matrix2d(sig_inv.asDiagonal());
}
*/

