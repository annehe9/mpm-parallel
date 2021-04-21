#include "eigen3/Eigen/Dense"
using namespace Eigen;

class cudaMPM {
public:
	// Particle representation
	struct Particle
	{
		Particle(double _x, double _y) : x(_x, _y), v(0.0, 0.0), F(Matrix2d::Identity()), C(Matrix2d::Zero()), Jp(1.0) {}
		Vector2d x, v; //position and velocity
		Matrix2d F, C; //deformation gradient, APIC momentum
		double Jp; //determinant of deformation gradient, which is volume
	};

	cudaMPM();
	void setup();
	void addParticles(double xcenter, double ycenter);
	void P2G(void);
	void UpdateGridVelocity(void);
	void G2P(void);
	void Update(void);

private:
	static int NUM_PARTICLES;
	static int BLOCK_PARTICLES;
	const static int GRID_RES = 80;
	vector<Particle> particles;
	Vector3d grid[GRID_RES + 1][GRID_RES + 1];

	vector<Particle> cudaDeviceParticles;
	Vector3d cudaDeviceGrid[GRID_RES + 1][GRID_RES + 1];
};
