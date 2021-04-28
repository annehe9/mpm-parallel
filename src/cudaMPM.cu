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

#include "eigen3/Eigen/Dense"
using namespace Eigen;

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaMPM.h"
#include "helper.h"

// Params we need outside of kernels or that need to be modified
struct GlobalConstants {
	Particle *particles;
	Vector3d *grid_block;
        int *NUM_PARTICLES_SET;
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


// GPU function
__global__ void P2G(void)
{
	int iterations = (*cuConstParams.NUM_PARTICLES_SET + BLOCKSIZE - 1) / BLOCKSIZE; // TODO
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
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
				Vector2d dpos = (Vector2d(i, j) - fx) * DX;
				// Translational momentum
				Vector3d mass_x_velocity(p.v.x() * MASS, p.v.y() * MASS, MASS);
				Vector2d tmp = affine * dpos;
				*(Vector3d*)(&cuConstParams.grid_block[
                                        (base_coord.x() + i) * GRID_RES + (base_coord.y() + j)
                                ]) += (
                                        w[i].x() * w[j].y() * (mass_x_velocity + Vector3d(tmp.x(), tmp.y(), 0))
                                );
			}
		}
	}
}

// GPU function
__global__ void UpdateGridVelocity(void) {
	int iterations = BLOCKSIZE; //(NUM_CELLS + BLOCKSIZE - 1) / BLOCKSIZE;
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
		int i = index / GRID_RES;
		int j = index % GRID_RES;
		Vector3d& g = cuConstParams.grid_block[i * GRID_RES + j];
		if (g[2] > 0) {
			// Normalize by mass
			g /= g[2];
			// Gravity
			g += DT * Vector3d(0, -200, 0);

			// boundary thickness
			double boundary = 0.05;
			// Node coordinates
			double x = (double)i / GRID_RES;
			double y = (double)j / GRID_RES;

			// Sticky boundary
			if (x < boundary || x > 1 - boundary) {
				g[0] = 0.0;
			}
			// Separate boundary
			if (y < boundary || y > 1 - boundary) {
				g[1] = 0.0;
			}
		}
	}
}

// GPU function
__global__ void G2P(void)
{
	int iterations = (*cuConstParams.NUM_PARTICLES_SET + BLOCKSIZE - 1) / BLOCKSIZE; // TODO
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
		Particle& p = cuConstParams.particles[index];
		// element-wise floor
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

		p.C = Matrix2d::Zero();
		p.v = Vector2d::Zero();

		// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
		// see APIC paper (https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
		// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
		// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				Vector2d dpos = (Vector2d(i, j) - fx);
				Vector3d curr = cuConstParams.grid_block[
                                        (base_coord.x() + i) * GRID_RES + (base_coord.y() + j)
                                ];
				Vector2d grid_v(curr.x(), curr.y());
				double weight = w[i].x() * w[j].y();
				// Velocity
				p.v += weight * grid_v;
				// APIC C, outer product of weighted velocity and dist, paper equation 10
				p.C += 4 * INV_DX * ((weight * grid_v) * dpos.transpose());
			}
		}

		// Advection
		//double tempy = p.x.y();
		p.x += DT * p.v;

		// MLS-MPM F-update eqn 17
		Matrix2d F = (Matrix2d::Identity() + DT * p.C) * p.F;

        SVDResults *R = (SVDResults *) malloc(sizeof(SVDResults));
		SolveJacobiSVD(F, R);
		Matrix2d svd_u = R->U;
		Matrix2d svd_v = R->V;
		// Snow Plasticity
        Matrix2d sig = R->singularValues;
		sig(0, 0) = min(max(sig(0, 0), 1.0 - 2.5e-2), 1.0 + 7.5e-3);
		sig(1, 1) = min(max(sig(1, 1), 1.0 - 2.5e-2), 1.0 + 7.5e-3);

		double oldJ = determinant(F);
		F = svd_u * sig * svd_v.transpose();

		double Jp_new = min(max(p.Jp * oldJ / determinant(F), 0.6), 20.0);

		p.Jp = Jp_new;
		p.F = F;

                /*
                if (index == 0) {
                        printf("Particle 0 pos (GPU): %f %f\n", p.x[0], p.x[1]);
                }
                */
	}
}

// CPU
void cudaMPM::Update(void)
{
        // parallelization over particles
        dim3 blockDim(BLOCKSIZE, 1);
        dim3 gridDim(
                (NUM_PARTICLES + blockDim.x - 1) / blockDim.x,
                (1 + blockDim.y - 1) / blockDim.y
        );
        // parallelization over grid
	dim3 blockDimG(BLOCKSIDE, BLOCKSIDE);
        dim3 gridDimG(1, 1);

        int grid_block_side = (OFFSIZE + BLOCKSIDE - 1) / BLOCKSIDE;

        int index, i, j, m, n;

        // starting point of block (wrap around for neighbors)
        int grid_block_x = 0;
        int grid_block_y = 0;
        int nsize = 0;

        memset(grid, 0, NUM_CELLS);

        // process grids and particles in blocks
        for (i = 0; i < grid_block_side; i++) {
                for (j = 0; j < grid_block_side; j++) {
                        index = i * grid_block_side + j;

                        *pSize = retrieve_block(index, particles, particle_set);
                        gridDim.x = (*pSize + blockDim.x - 1) / blockDim.x;

                        // copy particle data to GPU,
                        cudaMemcpy(
                                (void**)&cudaDeviceParticles,
                                (void**)particle_set,
                                sizeof(Particle) * *pSize, 
                                cudaMemcpyHostToDevice
                        );

                        // copy grid block to grid_block >> TODO
                        for (m = 0; m < BLOCKSIDE; m++) {
                                for (n = 0; n < BLOCKSIDE; n++) {
                                        grid_block[m][n] =
                                                grid[m + grid_block_x][n + grid_block_y];
                                }
                        }

                        // copy grid data to GPU to get neighbors
                        cudaMemcpy(
                                cudaDeviceBlock, 
                                (void **)grid_block,
                                sizeof(Vector3d) * BLOCKSIZE, 
                                cudaMemcpyHostToDevice
                        );

                        P2G<<<gridDim, blockDim>>>();
                        UpdateGridVelocity<<<gridDimG, blockDimG>>>();
                        G2P<<<gridDim, blockDim>>>();

                        // copy particle data back to CPU
                        cudaMemcpy(
                                (void**)(particle_next + nsize),
                                (void**)&cudaDeviceParticles,
                                sizeof(Particle) * *pSize, 
                                cudaMemcpyDeviceToHost
                        );
                        nsize += *pSize;

                        // copy grid data back to CPU
                        cudaMemcpy(
                                (void **)grid_block,
                                cudaDeviceBlock, 
                                sizeof(Vector3d) * BLOCKSIZE, 
                                cudaMemcpyDeviceToHost
                        );

                        // copy grid_block to grid block
                        for (m = 0; m < BLOCKSIDE; m++) {
                                for (n = 0; n < BLOCKSIDE; n++) {
                                        grid[m + grid_block_x][n + grid_block_y]
                                                = grid_block[m][n];
                                }
                        }

                        // starting point of block (wrap around for neighbors)
                        grid_block_x += OFFSIDE;
                        grid_block_y += OFFSIDE;
                }

                // swap particle buffers by changing pointers
                // variable should contain address to first element
                Particle *temp = particles;
                Particle *particles = particle_next;
                particle_next = temp;

                // Update blocks each particle is in.
                map_particles(particles, NUM_PARTICLES);
        }
}

// CPU
cudaMPM::cudaMPM() {
	NUM_PARTICLES = 0;

        particles = (Particle *) malloc(sizeof(Particle) * MAX_PARTICLES);
        grid = (Vector3d *) malloc(sizeof(Vector3d *) * NUM_CELLS);
        grid_block = (Vector3d *) malloc(sizeof(Vector3d *) * BLOCKSIZE);
}

// CPU, prep GPU
void cudaMPM::setup(void)
{

        // CPU particles
	addParticles(0.55, 0.45);
	addParticles(0.45, 0.65);
	addParticles(0.55, 0.85);

	cout << "initializing mpm with " << NUM_PARTICLES / 3 << " particles" << endl;

        particle_set = (Particle *) malloc(sizeof(Particle) * NUM_PARTICLES);
        particle_next = (Particle *) malloc(sizeof(Particle) * NUM_PARTICLES);

        // GPU memory
	cudaMalloc(
                (void**)cudaDeviceParticles, 
                sizeof(Particle) * NUM_PARTICLES
        );

        // outer ring is for neighborhood 
	cudaMalloc(
                (void**)cudaDeviceBlock, 
                sizeof(Vector3d) * BLOCKSIZE
        );

        // copy initial set of particles, CPU->GPU 
	cudaMemcpy(
                (void**)cudaDeviceParticles,
                (void**)particles,
                sizeof(Particle) * NUM_PARTICLES, 
                cudaMemcpyHostToDevice
        );

        // begin data structure
        map_setup();
        map_particles(particles, NUM_PARTICLES);

        // get global cuda params
        params.NUM_PARTICLES = NUM_PARTICLES;
        params.NUM_PARTICLES_SET = pSize;
	params.particles = cudaDeviceParticles;
	params.grid_block = cudaDeviceBlock;

        cudaMemcpyToSymbol(
                cuConstParams,
                &params,
                sizeof(GlobalConstants)
        );
}
