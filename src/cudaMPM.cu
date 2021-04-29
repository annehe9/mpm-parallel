#if __APPLE__
#include <GLUT/glut.h>
#else
//#include <windows.h>
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaMPM.h"
//#include "helper.h"


struct SVDResults {
        Matrix2d V;
        Matrix2d U;
        Matrix2d singularValues;
};

__device__ double determinant(Matrix2d M) {
        return M(0,0) * M(1,1) - M(0,1) * M(1,0);
}

// https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
// watch out if M is identity
__device__ void SolveJacobiSVD(Matrix2d M, SVDResults *R) {
       R->V = Matrix2d::Zero();
       R->U = Matrix2d::Zero();
       R->singularValues = Matrix2d::Zero();

       // handle special identity matrix case
       if (M(0, 0) == 1
           && M(0, 1) == 0
           && M(1, 0) == 0
           && M(1, 1) == 1) {
               R->V = Matrix2d::Identity();
               R->U = Matrix2d::Identity();
               R->singularValues = Matrix2d::Identity();
               return;
       //}

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



// Params we need outside of kernels or that need to be modified
struct GlobalConstants {
	Particle *particles;
	Vector3d *grid;
        int NUM_PARTICLES;
};

GlobalConstants params;

__constant__ GlobalConstants cuConstParams;

// CPU function
void cudaMPM::addParticles(double xcenter, double ycenter)
{
	for (int i = 0; i < BLOCK_PARTICLES; i++) {
                particles[i + NUM_PARTICLES] = 
                        Particle((((double)rand() / 
                        (double)RAND_MAX) * 2 - 1) * 0.08 + xcenter, 
                        (((double)rand() / (double)RAND_MAX) * 2 - 1) 
                        * 0.08 + ycenter);
	}
	NUM_PARTICLES += BLOCK_PARTICLES;
}

// GPU function, changes grid, not particle
__global__ void P2G(void)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cuConstParams.NUM_PARTICLES) return; 

        const Particle& p = cuConstParams.particles[index];
        Vector2i base_coord = (p.x * INV_DX - Vector2d(0.5, 0.5)).cast<int>();
        Vector2d fx = p.x * INV_DX - base_coord.cast<double>();

        // Quadratic kernels [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 123, with x=fx, fx-1,fx-2]
        Vector2d onehalf(1.5, 1.5); //have to make these so eigen doesn't give me errors
        Vector2d one(1.0, 1.0);
        Vector2d half(0.5, 0.5);
        Vector2d threequarters(0.75, 0.75);
        Vector2d tmpa = onehalf - fx;
        Vector2d tmpb = fx - one;
        Vector2d tmpc = fx - half;
        Vector2d w[3] = {
          0.5 * (tmpa.cwiseProduct(tmpa)),
          threequarters - (tmpb.cwiseProduct(tmpb)),
          0.5 * (tmpc.cwiseProduct(tmpc))
        };

        // Snow
        // Compute current Lamé parameters [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 87]
        double e = exp(HARD * (1.0 - p.Jp));
        double mu = MU_0 * e;
        double lambda = LAMBDA_0 * e;

        // Current volume
        double J = determinant(p.F);

        // Polar decomposition for fixed corotated model, https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf paragraph after Eqn. 45
        SVDResults *R = (SVDResults *) malloc(sizeof(SVDResults));
        SolveJacobiSVD(p.F, R);
        Matrix2d r = R->U * R->V.transpose();
        Matrix2d s = R->V * R->singularValues * R->V.transpose();

        /*
        printf("F: (%f, %f, %f, %f)\n => S: (%f, %f, %f, %f)\nV: (%f, %f, %f, %f)\nU: (%f, %f, %f, %f)\n",
                p.F(0, 0), p.F(0, 1), p.F(1, 0), p.F(1, 1),                
                R->singularValues(0, 0), R->singularValues(0, 1), R->singularValues(1, 0), R->singularValues(1, 1),
                R->V(0, 0), R->V(0, 1), R->V(1, 0), R->V(1, 1),
                R->U(0, 0), R->U(0, 1), R->U(1, 0), R->U(1, 1) 
        );
        */

        // [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Paragraph after Eqn. 176]
        double Dinv = 4 * INV_DX * INV_DX;
        // [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 52]
        Matrix2d PF_0 = (2 * mu) * (p.F - r) * p.F.transpose();
        double pf1tmp = (lambda * (J - 1) * J);
        Matrix2d PF_1;
        PF_1 << pf1tmp, pf1tmp,
                pf1tmp, pf1tmp;
        Matrix2d PF = PF_0 + PF_1;

        // Cauchy stress times dt and inv_dx
        Matrix2d stress = -(DT * VOL) * (Dinv * PF);

        // Fused APIC momentum + MLS-MPM stress contribution
        // See https://yzhu.io/publication/mpmmls2018siggraph/paper.pdf
        // Eqn 29
        Matrix2d affine = stress + MASS * p.C;

        // P2G
        for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                        int x = base_coord.x() + i;
                        int y = base_coord.y() + j;
                        Vector2d dpos = (Vector2d(i, j) - fx) * DX;
                        // Translational momentum
                        Vector3d mass_x_velocity(p.v.x() * MASS, p.v.y() * MASS, MASS);
                        Vector2d tmp = affine * dpos;
                        if (x * GRID_RES + y < 0 || x * GRID_RES + y >= NUM_CELLS)
                                printf("P2G grid access %d (%d, %d), (%f, %f)\n", x * GRID_RES + y, x, y,
                                        cuConstParams.particles[index].x(0),
                                        cuConstParams.particles[index].x(1)
                                );
                        cuConstParams.grid[x * GRID_RES + y] += (
                                w[i].x() * w[j].y() * (mass_x_velocity + Vector3d(tmp.x(), tmp.y(), 0))
                        );
                }
        }
}

// GPU function
__global__ void UpdateGridVelocity(void) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= GRID_RES || j >= GRID_RES) return;

        int index = i * GRID_RES + j;
        if (index < 0 || index >= NUM_CELLS)
                printf("UpdateGridVelocity grid access %d\n", index);
        //Vector3d& g = cuConstParams.grid[index];
        // block to grid
        if (cuConstParams.grid[index][2] > 0) {
                // Normalize by mass
                cuConstParams.grid[index] /= cuConstParams.grid[index][2];
                // Gravity
                cuConstParams.grid[index] += DT * Vector3d(0, -200, 0);

                // boundary thickness
                double boundary = 0.05;
                // Node coordinates
                double x = (double)i / GRID_RES;
                double y = (double)j / GRID_RES;

                // Sticky boundary
                if (x < boundary || x > 1 - boundary) {
                        cuConstParams.grid[index][0] = 0.0;
                }
                // Separate boundary
                if (y < boundary || y > 1 - boundary) {
                        cuConstParams.grid[index][1] = 0.0;
                }
        }
        /*
        printf("UpdateGridVelocity: (%d, %d) => (%f, %f, %f)\n",
                i, j, cuConstParams.grid[i * GRID_RES + j][0], cuConstParams.grid[i * GRID_RES + j][1], cuConstParams.grid[i * GRID_RES + j][2]
        );
        */
}

// GPU function, changes particle, not grid
__global__ void G2P(void)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cuConstParams.NUM_PARTICLES) return; 

        //Particle& p = cuConstParams.particles[index];
        // element-wise floor
        Vector2i base_coord = (cuConstParams.particles[index].x * INV_DX - Vector2d(0.5, 0.5)).cast<int>();
        Vector2d fx = cuConstParams.particles[index].x * INV_DX - base_coord.cast<double>();

        // Quadratic kernels [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 123, with x=fx, fx-1,fx-2]
        Vector2d onehalf(1.5, 1.5); //have to make these so eigen doesn't give me errors
        Vector2d one(1.0, 1.0);
        Vector2d half(0.5, 0.5);
        Vector2d threequarters(0.75, 0.75);
        Vector2d tmpa = onehalf - fx;
        Vector2d tmpb = fx - one;
        Vector2d tmpc = fx - half;
        Vector2d w[3] = {
          0.5 * (tmpa.cwiseProduct(tmpa)),
          threequarters - (tmpb.cwiseProduct(tmpb)),
          0.5 * (tmpc.cwiseProduct(tmpc))
        };

        cuConstParams.particles[index].C = Matrix2d::Zero();
        cuConstParams.particles[index].v = Vector2d::Zero();

        // constructing affine per-particle momentum matrix from APIC / MLS-MPM.
        // see APIC paper (https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
        // below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
        // where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions

        for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                        // grid to block
                        int x = base_coord.x() + i;
                        int y = base_coord.y() + j; 
                        Vector2d dpos = (Vector2d(i, j) - fx);
                        if (x * GRID_RES + y < 0 || x * GRID_RES + y >= NUM_CELLS)
                                printf("G2P grid access %d (%d, %d) (%f, %f)\n", x * GRID_RES + y, x, y,
                                        cuConstParams.particles[index].x(0),
                                        cuConstParams.particles[index].x(1)
                                );
                        Vector3d curr = cuConstParams.grid[x * GRID_RES + y];
                        Vector2d grid_v(curr.x(), curr.y());
                        double weight = w[i].x() * w[j].y();
                        // Velocity
                        cuConstParams.particles[index].v += weight * grid_v;
                        // APIC C, outer product of weighted velocity and dist, paper equation 10
                        cuConstParams.particles[index].C += 4 * INV_DX * ((weight * grid_v) * dpos.transpose());
                }
        }

        // Advection
        // UPDATE PARTICLE POSITION 
        //double tempy = p.x.y();
        Vector2d temp = cuConstParams.particles[index].x;
        cuConstParams.particles[index].x += DT * cuConstParams.particles[index].v;

        /*
        printf("G2P 1 (%d) => F:(%f, %f, %f, %f) C: (%f, %f, %f, %f) Jp:%f\n", 
               index,
               cuConstParams.particles[index].F(0, 0), cuConstParams.particles[index].F(0, 1), cuConstParams.particles[index].F(1, 0), cuConstParams.particles[index].F(1, 1),
               cuConstParams.particles[index].C(0, 0), cuConstParams.particles[index].C(0, 1), cuConstParams.particles[index].C(1, 0), cuConstParams.particles[index].C(1, 1),
               cuConstParams.particles[index].Jp
        );
        */

        // MLS-MPM F-update eqn 17
        Matrix2d F = (Matrix2d::Identity() + DT * cuConstParams.particles[index].C) * cuConstParams.particles[index].F;

        SVDResults *R = (SVDResults *) malloc(sizeof(SVDResults));
        SolveJacobiSVD(F, R);
        Matrix2d svd_u = R->U;
        Matrix2d svd_v = R->V;
        // Snow Plasticity
        Matrix2d sig = R->singularValues;

        /*
        printf("G2P 2 (%d) => F: (%f, %f, %f, %f)\n => S: (%f, %f, %f, %f)\nV: (%f, %f, %f, %f)\nU: (%f, %f, %f, %f)\n",
                index,
                F(0, 0), F(0, 1), F(1, 0), F(1, 1),                
                R->singularValues(0, 0), R->singularValues(0, 1), R->singularValues(1, 0), R->singularValues(1, 1),
                R->V(0, 0), R->V(0, 1), R->V(1, 0), R->V(1, 1),
                R->U(0, 0), R->U(0, 1), R->U(1, 0), R->U(1, 1) 
        );
        */

        sig(0, 0) = min(max(sig(0, 0), 1.0 - 2.5e-2), 1.0 + 7.5e-3);
        sig(1, 1) = min(max(sig(1, 1), 1.0 - 2.5e-2), 1.0 + 7.5e-3);

        double oldJ = determinant(F);
        F = svd_u * sig * svd_v.transpose();

        /*
        printf("G2P 3 (%d) => xF:(%f, %f, %f, %f) oldJp:%f\n", 
                F(0, 0), F(0, 1), F(1, 0), F(1, 1), oldJ
        );
        */

        double Jp_new = min(max(cuConstParams.particles[index].Jp * oldJ / determinant(F), 0.6), 20.0);

        cuConstParams.particles[index].Jp = Jp_new;
        cuConstParams.particles[index].F = F;

        /*
        printf("G2P (%d) => x:(%f, %f) v:(%f, %f) F:(%f, %f, %f, %f) C: (%f, %f, %f, %f) Jp:%f\n", 
               index,
               cuConstParams.particles[index].x(0), cuConstParams.particles[index].x(1),
               cuConstParams.particles[index].v(0), cuConstParams.particles[index].v(1),
               cuConstParams.particles[index].F(0, 0), cuConstParams.particles[index].F(0, 1), cuConstParams.particles[index].F(1, 0), cuConstParams.particles[index].F(1, 1),
               cuConstParams.particles[index].C(0, 0), cuConstParams.particles[index].C(0, 1), cuConstParams.particles[index].C(1, 0), cuConstParams.particles[index].C(1, 1),
               cuConstParams.particles[index].Jp
        );
        */
}

// CPU
void cudaMPM::Update(bool debug)
{
        // zero-out cuda grid
        cudaMemset(
                cudaDeviceGrid, 
                0, 
                sizeof(Vector3d) * NUM_CELLS
        );

        // copy particle data to GPU
        cudaMemcpy(
                (void*)cudaDeviceParticles,
                (void*)particles,
                sizeof(Particle) * NUM_PARTICLES, 
                cudaMemcpyHostToDevice
        );

        /*
        if (debug) {
                for (int a = 0; a < NUM_PARTICLES; a++) {
                        printf("particles: (%f, %f), (%f, %f)\n", 
                                particles[a].x(0), particles[a].x(1),
                                particles[a].v(0), particles[a].v(1)
                        );
                }
        }
        */

        // parallelization over particles
        dim3 blockDim(BLOCKSIZE, 1);
        dim3 gridDim(
                (NUM_PARTICLES + blockDim.x - 1) / blockDim.x, 1
        );
        P2G<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();

        /*
        cudaMemcpy(
                (void*)grid,
                (void*)cudaDeviceGrid,
                sizeof(Vector3d) * NUM_CELLS, 
                cudaMemcpyDeviceToHost
        );
        for (int i = 0; i < GRID_RES; i++) {
                for (int j = 0; j < GRID_RES; j++) {
                        printf("P2G: (%d, %d) => (%f, %f, %f)\n",
                                i, j, grid[i * GRID_RES + j][0], grid[i * GRID_RES + j][1], grid[i * GRID_RES + j][2]
                        );
                }
        }
        */

        // parallelization over grid
        blockDim.x = BLOCKSIDE;
        blockDim.y = BLOCKSIDE;
        gridDim.x = (GRID_RES + blockDim.x - 1) / blockDim.x;
        gridDim.y = (GRID_RES + blockDim.y - 1) / blockDim.y;
        UpdateGridVelocity<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();

        // parallelization over particles
        blockDim.x = BLOCKSIZE;
        blockDim.y = 1;
        gridDim.x = (NUM_PARTICLES + blockDim.x - 1) / blockDim.x;
        gridDim.y = 1;
        G2P<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();

        // copy particle data back to CPU
        cudaError_t err = cudaMemcpy(
                (void*)particles,
                (void*)cudaDeviceParticles,
                sizeof(Particle) * NUM_PARTICLES, 
                cudaMemcpyDeviceToHost
        );
        printf("GPU->CPU COPY ERROR: %d\n", err);

        /*
        if (debug) {
                for (int a = 0; a < NUM_PARTICLES; a++) {
                        printf(">particles: (%f, %f), (%f, %f)\n", 
                                particles[a].x(0), particles[a].x(1),
                                particles[a].v(0), particles[a].v(1)
                        );
                }
        }
        */

        /*
        // copy grid data back to GPU to see
        cudaMemcpy(
                (void*)grid,
                (void*)cudaDeviceGrid,
                sizeof(Vector3d) * NUM_CELLS, 
                cudaMemcpyDeviceToHost
        );
        for (int a = 0; a < GRID_RES; a++) {
                for (int b = 0; b < GRID_RES; b++) {
                        printf("(%f, %f, %f), ", 
                                grid[a * GRID_RES + b](0),
                                grid[a * GRID_RES + b](1),
                                grid[a * GRID_RES + b](2)
                        );
                }
                printf("\n");
        }
        printf("\n\n");
        */
}

// CPU
cudaMPM::cudaMPM() {
	NUM_PARTICLES = 0;

        particles = (Particle *) malloc(sizeof(Particle) * MAX_PARTICLES);
        grid = (Vector3d *) malloc(sizeof(Vector3d) * NUM_CELLS);
}

// CPU, prep GPU
void cudaMPM::setup(void)
{
        // CPU particles
	addParticles(0.55, 0.45);
	addParticles(0.45, 0.65);
	addParticles(0.55, 0.85);

        /*
        int NUM_PARTICLES = 3;
        particles[0] = Particle(0.483769, 0.854159);
        particles[1] = Particle(0.525823, 0.912437);
        particles[2] = Particle(0.480095, 0.843232);
        */

	cout << "initializing mpm with " << NUM_PARTICLES << " particles" << endl;

        // GPU memory
	cudaMalloc(
                (void**)&cudaDeviceParticles, 
                sizeof(Particle) * NUM_PARTICLES
        );

	cudaMalloc(
                (void**)&cudaDeviceGrid, 
                sizeof(Vector3d) * NUM_CELLS
        );

        // get global cuda params
        params.NUM_PARTICLES = NUM_PARTICLES;
	params.particles = cudaDeviceParticles;
	params.grid = cudaDeviceGrid;

        cudaMemcpyToSymbol(
                cuConstParams,
                &params,
                sizeof(GlobalConstants)
        );
}
