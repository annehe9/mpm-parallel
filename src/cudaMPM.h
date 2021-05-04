#include "eigen3/Eigen/Dense"
using namespace Eigen;

#include "mapping.h"


class cudaMPM {
public:
        int NUM_PARTICLES;

	Particle *particles;
	Vector3d *grid;
        int *assignment;

	Particle *cudaDeviceParticles;
	Vector3d *cudaDeviceGrid;
        int *cudaDeviceAssignment;

	cudaMPM();
	void setup();
	void addParticles(double xcenter, double ycenter);
        void CPU_SVD_P2G();
        void CPU_SVD_G2P();
	void Update();
};
