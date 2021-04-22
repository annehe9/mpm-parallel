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

#define BLOCKSIDE 32
#define BLOCKSIZE ((BLOCKSIDE) * (BLOCKSIDE))

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// References:
// https://github.com/yuanming-hu/taichi_mpm/blob/master/mls-mpm88-explained.cpp
// https://lucasschuermann.com/writing/implementing-sph-in-2d for visualization

// Granularity
#define BLOCK_PARTICLES 1000		// number of particles added in a block
#define GRID_RES 80
#define NUM_CELLS GRID_RES * GRID_RES	// number of cells in the grid
#define DT 0.00001			// integration timestep
#define DX 1.0 / GRID_RES
#define INV_DX 1.0 / DX

// Simulation params
#define MASS 1.0					// mass of one particle
#define VOL 1.0					// volume of one particle
#define HARD 10.0					// snow hardening factor
#define E 10000				// Young's Modulus, resistance to fracture
#define NU 0.2					// Poisson ratio

// Initial Lame params
#define MU_0 E / (2 * (1 + NU))
#define LAMBDA_0 (E * NU) / ((1 + NU) * (1 - 2 * NU))

// Params we need outside of kernels or that need to be modified
struct GlobalConstants {
	int NUM_PARTICLES;					// keeps track of current number of particles
	// Data structures
	cudaMPM::Particle* particles;
	// Vector3: [velocity_x, velocity_y, mass]
	Vector3d** grid;
};

GlobalConstants params;

__constant__ GlobalConstants cuConstParams;

// CPU function
void cudaMPM::addParticles(double xcenter, double ycenter)
{
	for (int i = 0; i < BLOCK_PARTICLES; i++) {
		particles.push_back(cudaMPM::Particle((((double)rand() / (double)RAND_MAX) * 2 - 1) * 0.08 + xcenter, (((double)rand() / (double)RAND_MAX) * 2 - 1) * 0.08 + ycenter));
	}
	params.NUM_PARTICLES += BLOCK_PARTICLES;
}

// GPU function
__global__ void P2G(void)
{
	int iterations = (cuConstParams.NUM_PARTICLES + BLOCKSIZE - 1) / BLOCKSIZE;
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
		const cudaMPM::Particle& p = cuConstParams.particles[index];
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
		// Compute current Lam� parameters [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 87]
		double e = exp(HARD * (1.0 - p.Jp));
		double mu = MU_0 * e;
		double lambda = LAMBDA_0 * e;

		// Current volume
		double J = 0; // p.F.determinant(); // TODO: determinant

		// Polar decomposition for fixed corotated model, https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf paragraph after Eqn. 45
		//JacobiSVD<Matrix2d> svd(p.F, ComputeFullU | ComputeFullV); // TODO: JacobiSVD
		Matrix2d r(2, 2); //= svd.matrixU() * svd.matrixV().transpose(); // TODO: related to above
		//Matrix2d s = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();

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
				*(Vector3d*)(&cuConstParams.grid[base_coord.x() + i][base_coord.y() + j]) 
                                        += (
					        w[i].x() * w[j].y() * (mass_x_velocity + Vector3d(tmp.x(), tmp.y(), 0))
					);
			}
		}
	}
}

// GPU function
__global__ void UpdateGridVelocity(void) {
	int iterations = (NUM_CELLS + BLOCKSIZE - 1) / BLOCKSIZE;
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
		int i = index / GRID_RES;
		int j = index % GRID_RES;
		Vector3d& g = cuConstParams.grid[i][j];
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
	int iterations = (cuConstParams.NUM_PARTICLES + BLOCKSIZE - 1) / BLOCKSIZE;
	for (int i = 0; i < iterations; i++) {
		int index = blockIdx.x * blockDim.x + threadIdx.x + i;
		cudaMPM::Particle& p = cuConstParams.particles[index];
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
				Vector3d curr = cuConstParams.grid[base_coord.x() + i][base_coord.y() + j];
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
		//cout << "change" << tempy - p.x.y() << endl;
		//cout << "velocity " <<  p.v << endl;

		// MLS-MPM F-update eqn 17
		Matrix2d F = (Matrix2d::Identity() + DT * p.C) * p.F;

		//JacobiSVD<Matrix2d> svd(F, ComputeFullU | ComputeFullV); // TODO: JacobiSVD
		Matrix2d svd_u(2, 2); //svd.matrixU(); // TODO: ^
		Matrix2d svd_v(2, 2); //svd.matrixV(); // TODO: ^
                Vector2d sigvalues(2); //svd.singularValues().array().min(1.0f + 7.5e-3).max(1.0 - 2.5e-2); // TODO; may not need
		// Snow Plasticity
		Matrix2d sig = sigvalues.asDiagonal();

		double oldJ = 0; //F.determinant(); // TODO: determinant
		F = svd_u * sig * svd_v.transpose();

		double Jp_new = 0; //min(max(p.Jp * oldJ / F.determinant(), 0.6), 20.0); // TODO: determinant

		p.Jp = Jp_new;
		p.F = F;
	}
}

// CPU
void cudaMPM::Update(void)
{
	dim3 blockDim(BLOCKSIZE, 1);
	dim3 gridDim(
		(params.NUM_PARTICLES + blockDim.x - 1) / blockDim.x,
		(1 + blockDim.y - 1) / blockDim.y
	);
	dim3 blockDimG(BLOCKSIDE, BLOCKSIDE);
	dim3 gridDimG(
		(GRID_RES + BLOCKSIDE - 1) / BLOCKSIDE,
		(GRID_RES + BLOCKSIDE - 1) / BLOCKSIDE
	);

        // 0 out CPU and GPU grids
	memset(grid, 0, sizeof(grid));
	cudaMemset(
                cudaDeviceGrid, 
                0, 
                (GRID_RES) * (GRID_RES) * sizeof(Vector3d)
        );
        // use same particle data on GPU as from before

	P2G<<<gridDim, blockDim>>>();
	UpdateGridVelocity<<<gridDimG, blockDimG>>>();
	G2P<<<gridDim, blockDim>>>();

        // copy particle/grid data back to CPU
	cudaMemcpy(
                &particles[0], 
                &cudaDeviceParticles[0], 
                sizeof(cudaMPM::Particle) * params.NUM_PARTICLES, 
                cudaMemcpyDeviceToHost
        );
        // copy grid data back to CPU
	cudaMemcpy(
                grid, 
                cudaDeviceGrid, 
                sizeof(Vector3d) * (GRID_RES) * (GRID_RES), 
                cudaMemcpyDeviceToHost
        );
}

// CPU
cudaMPM::cudaMPM() {
        // CPU grid
	params.NUM_PARTICLES = 0;
	for (int i = 0; i < GRID_RES; i++){
		grid[i] = (Vector3d*)malloc(GRID_RES * sizeof(Vector3d));
	}
}

// CPU, prep GPU
void cudaMPM::setup(void)
{
        // CPU particles
	addParticles(0.55, 0.45);
	addParticles(0.45, 0.65);
	addParticles(0.55, 0.85);

        // GPU memory
	cudaMalloc(
                (void**)&*cudaDeviceParticles.begin(), 
                sizeof(cudaMPM::Particle) * params.NUM_PARTICLES
        );
	cudaMalloc(
                (void**)&cudaDeviceGrid, 
                sizeof(double) * 3 * (GRID_RES) * (GRID_RES)
        );

        // copy initial set of particles, CPU->GPU 
	cudaMemcpy(
                &cudaDeviceParticles[0], 
                &particles[0], 
                sizeof(cudaMPM::Particle) * params.NUM_PARTICLES, 
                cudaMemcpyHostToDevice
        );

        // copy initial grid, CPU -> GPU
	for (int i = 0; i < GRID_RES; i++){
		cudaMemcpy(
                        cudaDeviceGrid[i], 
                        grid[i], 
                        sizeof(Vector3d) * (GRID_RES), 
                        cudaMemcpyHostToDevice
                );
	}

	cout << "initializing mpm with " << params.NUM_PARTICLES/3 << " particles" << endl;

	params.particles = &cudaDeviceParticles[0];
	params.grid = cudaDeviceGrid;

        cudaMemcpyToSymbol(
                cuConstParams,
                &params,
                sizeof(GlobalConstants)
        );
}