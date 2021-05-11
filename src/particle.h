#include "eigen3/Eigen/Dense"
using namespace Eigen;

#define DEBUG 0

#define MAX_PARTICLES 100000

#define BLOCKSIDE 16
#define BLOCKSIZE ((BLOCKSIDE) * (BLOCKSIDE))
// size of each block
#define OFFSIDE ((BLOCKSIDE) - 2)
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
#define BLOCK_PARTICLES 2000		// number of particles added in a block
#define GRID_RES 40
#define TRUE_GRID_RES ((GRID_RES) - 2)
#define NUM_CELLS ((GRID_RES) * (GRID_RES))	// number of cells in the grid
#define NUM_TRUE_CELLS ((TRUE_GRID_RES) * (TRUE_GRID_RES))
#define DT 0.00001			// integration timestep
#define DX (1.0 / (GRID_RES))
#define INV_DX (1.0 / (DX))

// Render params
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

//#define MAX_PARTICLES_PER_CELL (((WINDOW_WIDTH) / (GRID_RES)) * ((WINDOW_HEIGHT) / (GRID_RES)))
// based on observations
//#define MAX_PARTICLES_PER_CELL 15 /* 500-60, 500-80, 1000-80 */
//#define MAX_PARTICLES_PER_CELL 20 /* 500-40, 1000-60 */
//#define MAX_PARTICLES_PER_CELL 25 /* 2000-80 */
//#define MAX_PARTICLES_PER_CELL 35 /* 2000-60 */
//#define MAX_PARTICLES_PER_CELL 45 /* 1000-40 */
#define MAX_PARTICLES_PER_CELL 65 /* 2000-40 */

#define GRID_BLOCK_SIDE (((TRUE_GRID_RES) + (OFFSIDE) - 1) / (OFFSIDE))

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
        Particle(double _x, double _y) : F(Matrix2d::Identity()), C(Matrix2d::Zero()), x(_x, _y), v(0.0, 0.0), Jp(1.0) {}
        Matrix2d F, C; //deformation gradient, APIC momentum
        Matrix2d U, V, S; // associated with F's SVD computation
        Vector2d x, v; //position and velocity
        Vector3d grid_update[9];
        double Jp; //determinant of deformation gradient, which is volume
        double _padding[8]; // extra for padding to resolve sizeof issues, size is now 60 doubles
};
