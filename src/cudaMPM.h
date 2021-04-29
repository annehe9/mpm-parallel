#include "eigen3/Eigen/Dense"
using namespace Eigen;

#include "mapping.h"


class cudaMPM {
public:
        int NUM_PARTICLES;

	Particle *particles;
	Vector3d *grid;

	Particle *cudaDeviceParticles;
	Vector3d *cudaDeviceGrid;

	cudaMPM();
	void setup();
	void addParticles(double xcenter, double ycenter);
	void Update(bool debug);
};
