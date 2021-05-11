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

static int step = 0;

__device__ double determinant(Matrix2d M) {
        return M(0,0) * M(1,1) - M(0,1) * M(1,0);
}

// Params we need outside of kernels or that need to be modified
struct GlobalConstants {
	Particle *particles;
	Vector3d *grid;
        int *assignment;
        int NUM_PARTICLES;
};

GlobalConstants params;

__constant__ GlobalConstants cuConstParams;

static int rand_n = 0;
static int a1 = 10181801;
static int a2 = 20920473;
static int a3 = 12082087;

double pseudo_rand() {
        rand_n = (rand_n * a2 + a3) % a1;
        double ans = static_cast<double>(rand_n) / ((double)a1);
        if (ans < 0) ans *= -1;
        return ans;
}

// CPU function
void cudaMPM::addParticles(double xcenter, double ycenter)
{
	for (int i = 0; i < BLOCK_PARTICLES; i++) {
                double r1 = (double)rand() / ((double)RAND_MAX);
                double r2 = (double)rand() / ((double)RAND_MAX);
                //double r1 = pseudo_rand();
                //double r2 = pseudo_rand();
                particles[i + NUM_PARTICLES] = 
                        Particle(
                                (r1 * 2 - 1) * 0.08 + xcenter, 
                                (r2 * 2 - 1) * 0.08 + ycenter
                        );
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
        Matrix2d r = p.U * p.V.transpose();
        //Matrix2d s = p.V * p.S * p.V.transpose();

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
                        Vector3d a = w[i].x() * w[j].y() 
                                * (mass_x_velocity + Vector3d(tmp.x(), tmp.y(), 0));
                        cuConstParams.particles[index].grid_update[i * 3 + j] = a;
                }
        }
}

// GPU function
__global__ void P2GGrid(void) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i == 0 || i >= GRID_RES - 1 || j == 0 || j >= GRID_RES - 1) return;

        int index = i * GRID_RES + j;

        // perform P2G grid computation on internal area
        for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                        for (int k = 0; k < MAX_PARTICLES_PER_CELL; k++) {
                                // offset centered at (0,0)
                                int offset_x = (x-1);
                                int offset_y = (y-1);
                                int pdx = cuConstParams.assignment
                                        [((i + offset_x) * GRID_RES 
                                        + (j + offset_y)) 
                                        * MAX_PARTICLES_PER_CELL + k];
                                if (pdx == -1) break;

                                // invert offset
                                offset_x *= -1;
                                offset_y *= -1;
                                // center at (1,1)
                                offset_x += 1;
                                offset_y += 1;
                                cuConstParams.grid[index] += 
                                        cuConstParams.particles[pdx]
                                        .grid_update
                                        [offset_x * 3 + offset_y];
                        }
                }
        }
}

// GPU function
__global__ void UpdateGridVelocity(void) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= GRID_RES || j >= GRID_RES) return;

        int index = i * GRID_RES + j;

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
}

// GPU function, changes particle, not grid
__global__ void G2P(void)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cuConstParams.NUM_PARTICLES) return; 

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
                        Vector3d curr = cuConstParams.grid[x * GRID_RES + y];
                        Vector2d grid_v(curr.x(), curr.y());
                        double weight = w[i].x() * w[j].y();
                        // Velocity
                        cuConstParams.particles[index].v += weight * grid_v;
                        // APIC C, outer product of weighted velocity and dist, paper equation 10
                        Matrix2d a = 4 * INV_DX * ((weight * grid_v) * dpos.transpose());
                        cuConstParams.particles[index].C += a;
                }
        }

        // Advection
        // UPDATE PARTICLE POSITION 
        //double tempy = p.x.y();
        cuConstParams.particles[index].x += DT * cuConstParams.particles[index].v;

        // MLS-MPM F-update eqn 17
        Matrix2d F = (Matrix2d::Identity() + DT * cuConstParams.particles[index].C) * cuConstParams.particles[index].F;

        /* save intermediate for SVD computation on CPU */
        cuConstParams.particles[index].F = F;
}

// uses old F and Jp to make new F and Jp
void cudaMPM::CPU_SVD_G2P() {
        for (int i = 0; i < NUM_PARTICLES; i++) {
                JacobiSVD<Matrix2d> svd1(particles[i].F, ComputeFullU | ComputeFullV);
                Matrix2d svd_u = svd1.matrixU();
                Matrix2d svd_v = svd1.matrixV();

                // Snow Plasticity
                Matrix2d sig = Matrix2d(svd1.singularValues().asDiagonal());
                sig(0, 0) = min(max(sig(0, 0), 1.0 - 2.5e-2), 1.0 + 7.5e-3);
                sig(1, 1) = min(max(sig(1, 1), 1.0 - 2.5e-2), 1.0 + 7.5e-3);

                double oldJ = particles[i].F.determinant();
                Matrix2d F = svd_u * sig * svd_v.transpose();
                double Jp_new = min(max(particles[i].Jp * oldJ / F.determinant(), 0.6), 20.0);

                particles[i].Jp = Jp_new;
                particles[i].F = F;
       }
}

// compute SVD for F
void cudaMPM::CPU_SVD_P2G() {
        for (int i = 0; i < NUM_PARTICLES; i++) {
                JacobiSVD<Matrix2d> svd2(particles[i].F, ComputeFullU | ComputeFullV);
                particles[i].U = svd2.matrixU();
                particles[i].V = svd2.matrixV();
                particles[i].S = Matrix2d(svd2.singularValues().asDiagonal()); 
        }
}

double get_ms(struct timespec t) {
        return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

// CPU
void cudaMPM::Update()
{

        struct timespec t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;

        clock_gettime(CLOCK_REALTIME, &t1);
        // zero-out cuda grid
        cudaMemset(
                cudaDeviceGrid, 
                0, 
                sizeof(Vector3d) * NUM_CELLS
        );
        clock_gettime(CLOCK_REALTIME, &t2);
        // identify which grid cell each particle goes into
        memset(
                assignment,
                -1, // default
                sizeof(int) * NUM_CELLS * MAX_PARTICLES_PER_CELL
        );
        for (int i = 0; i < NUM_PARTICLES; i++) {
                Vector2i base_coord = (particles[i].x * INV_DX - Vector2d(0.5, 0.5)).cast<int>();
                int x = base_coord.x() + 1;
                int y = base_coord.y() + 1;
                // somewhat inefficient way of getting next available position
                int count = 0;
                while (assignment[(x * GRID_RES + y) * MAX_PARTICLES_PER_CELL + count] != -1) {
                        count++;
                } 
                // save particle index
                assignment[(x * GRID_RES + y) * MAX_PARTICLES_PER_CELL + count] = i;
        }
        clock_gettime(CLOCK_REALTIME, &t3);
        // copy assignment data to GPU
        cudaMemcpy(
                (void*)cudaDeviceAssignment,
                (void*)assignment,
                sizeof(int) * NUM_CELLS * MAX_PARTICLES_PER_CELL, 
                cudaMemcpyHostToDevice
        );
        clock_gettime(CLOCK_REALTIME, &t4);
        // perform SVD over particles on CPU
        CPU_SVD_P2G();
        clock_gettime(CLOCK_REALTIME, &t5);
        // copy particle data to GPU
        cudaMemcpy(
                (void*)cudaDeviceParticles,
                (void*)particles,
                sizeof(Particle) * NUM_PARTICLES, 
                cudaMemcpyHostToDevice
        );
        clock_gettime(CLOCK_REALTIME, &t6);
        // parallelization over particles
        dim3 blockDim(BLOCKSIZE, 1);
        dim3 gridDim(
                (NUM_PARTICLES + blockDim.x - 1) / blockDim.x, 1
        );
        P2G<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &t7);
        // parallelization over grid
        blockDim.x = BLOCKSIDE;
        blockDim.y = BLOCKSIDE;
        gridDim.x = (GRID_RES + blockDim.x - 1) / blockDim.x;
        gridDim.y = (GRID_RES + blockDim.y - 1) / blockDim.y;
        P2GGrid<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &t8);
        UpdateGridVelocity<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &t9);
        // parallelization over particles
        blockDim.x = BLOCKSIZE;
        blockDim.y = 1;
        gridDim.x = (NUM_PARTICLES + blockDim.x - 1) / blockDim.x;
        gridDim.y = 1;
        G2P<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &t10);
        // copy particle data back to CPU
        cudaError_t err = cudaMemcpy(
                (void*)particles,
                (void*)cudaDeviceParticles,
                sizeof(Particle) * NUM_PARTICLES, 
                cudaMemcpyDeviceToHost
        );
        clock_gettime(CLOCK_REALTIME, &t11);
        // perform SVD on particles
        CPU_SVD_G2P();
        clock_gettime(CLOCK_REALTIME, &t12);
        printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                step,                           // iteration
                get_ms(t2) - get_ms(t1),        // zero
                get_ms(t3) - get_ms(t2),        // assign
                get_ms(t4) - get_ms(t3),        // copy assign
                get_ms(t5) - get_ms(t4),        // P2G SVD
                get_ms(t6) - get_ms(t5),        // copy particle
                get_ms(t7) - get_ms(t6),        // P2G1
                get_ms(t8) - get_ms(t7),        // P2G2
                get_ms(t9) - get_ms(t8),        // UGV
                get_ms(t10) - get_ms(t9),       // G2P
                get_ms(t11) - get_ms(t10),      // copy particle
                get_ms(t12) - get_ms(t11),      // G2P SVD
                get_ms(t12) - get_ms(t1)       // total
        );
        step++;
}

// CPU
cudaMPM::cudaMPM() {
	NUM_PARTICLES = 0;

        particles = (Particle *) malloc(sizeof(Particle) * MAX_PARTICLES);
        grid = (Vector3d *) malloc(sizeof(Vector3d) * NUM_CELLS);
        assignment = (int *) malloc(sizeof(int) * NUM_CELLS * MAX_PARTICLES_PER_CELL);
}

// CPU, prep GPU
void cudaMPM::setup(void)
{
        // CPU particles
	addParticles(0.55, 0.45);
	addParticles(0.45, 0.65);
	addParticles(0.55, 0.85);

        cout << BLOCK_PARTICLES << " particles/block" << endl;
        cout << GRID_RES << " grid resolution" << endl;
        cout << MAX_PARTICLES_PER_CELL << " max particles per cell" << endl;

        // GPU memory
	cudaMalloc(
                (void**)&cudaDeviceParticles, 
                sizeof(Particle) * NUM_PARTICLES
        );

	cudaMalloc(
                (void**)&cudaDeviceGrid, 
                sizeof(Vector3d) * NUM_CELLS
        );
        cudaMalloc(
                (void**)&cudaDeviceAssignment,
                sizeof(int) * NUM_CELLS * MAX_PARTICLES_PER_CELL
        );

        // get global cuda params
        params.NUM_PARTICLES = NUM_PARTICLES;
	params.particles = cudaDeviceParticles;
	params.grid = cudaDeviceGrid;
        params.assignment = cudaDeviceAssignment;

        cudaMemcpyToSymbol(
                cuConstParams,
                &params,
                sizeof(GlobalConstants)
        );
}
