#include "eigen3/Eigen/Dense"
using namespace Eigen;

#define MAX_PARTICLES 100000

class cudaMPM {
public:
	const static int GRID_RES = 80;
        int NUM_PARTICLES;

	// Particle representation
	struct Particle
	{
		Particle(double _x, double _y) : x(_x, _y), v(0.0, 0.0), F(Matrix2d::Identity()), C(Matrix2d::Zero()), Jp(1.0) {}
		Vector2d x, v; //position and velocity
		Matrix2d F, C; //deformation gradient, APIC momentum
		double Jp; //determinant of deformation gradient, which is volume
	};

	Particle *particles;//vector<Particle> *particles;
	Vector3d *grid;

	Particle *cudaDeviceParticles;
	Vector3d *cudaDeviceGrid;

	cudaMPM();
	void setup();
	void addParticles(double xcenter, double ycenter);
	void Update(void);
};
