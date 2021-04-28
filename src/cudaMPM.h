#include "eigen3/Eigen/Dense"
using namespace Eigen;

#include "mapping.h"


class cudaMPM {
public:
        int NUM_PARTICLES;

	Particle *particles;
        Particle *particle_next;
        Particle *particle_set;
        int *pSize;

	Vector3d *grid;
        Vector3d *grid_block;

	Particle *cudaDeviceParticles;
	Vector3d *cudaDeviceBlock;

	cudaMPM();
	void setup();
	void addParticles(double xcenter, double ycenter);
	void Update(void);
};
