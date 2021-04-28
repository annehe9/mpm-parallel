#include "eigen3/Eigen/Dense"
using namespace Eigen;

#define MAX_PARTICLES 100000

#define BLOCKSIDE 32
#define BLOCKSIZE ((BLOCKSIDE) * (BLOCKSIDE))
// size of each block
#define OFFSIDE (BLOCKSIDE - 2)
#define OFFSIZE ((OFFSIDE) * (OFFSIDE))

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

// Particle representation
struct Particle
{
        Particle(double _x, double _y) : x(_x, _y), v(0.0, 0.0), F(Matrix2d::Identity()), C(Matrix2d::Zero()), Jp(1.0) {}
        Vector2d x, v; //position and velocity
        Matrix2d F, C; //deformation gradient, APIC momentum
        double Jp; //determinant of deformation gradient, which is volume
};
