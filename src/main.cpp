#if __APPLE__
#include <GLUT/glut.h>
#else
//#include <windows.h>
#include <GL/glut.h>
#endif

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cudaMPM.h"

// Granularity
const static int MAX_PARTICLES = 25000;
const static int BLOCK_PARTICLES = 1000;		// number of particles added in a block
int NUM_PARTICLES = 0;					// keeps track of current number of particles
const static int GRID_RES = 80;				// grid dim of one side
const static int NUM_CELLS = GRID_RES * GRID_RES;	// number of cells in the grid
const static double DT = 0.00001;			// integration timestep
const static double DX = 1.0 / GRID_RES;
const static double INV_DX = 1.0 / DX;

// Simulation params
const static double MASS = 1.0;					// mass of one particle
const static double VOL = 1.0;					// volume of one particle
const static double HARD = 10.0;					// snow hardening factor
const static double E = 10000;				// Young's Modulus, resistance to fracture
const static double NU = 0.2;					// Poisson ratio

// Initial Lame params
const static double MU_0 = E / (2 * (1 + NU));
const static double LAMBDA_0 = (E * NU) / ((1 + NU) * (1 - 2 * NU));

// Render params
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
//const static double VIEW_WIDTH = 1.5 * 800;
//const static double VIEW_HEIGHT = 1.5 * 600;

// Image output params
static int frame = 0;	// current image
static int step = 0;	// current simulation step
const static int FPS = 500; // fps of output video
const static int INV_FPS = (1.0 / DT) / FPS;

cudaMPM* solver;

void InitMPM(void)
{
	solver = new cudaMPM();
	solver->setup();
	cout << "initializing mpm with " << solver->NUM_PARTICLES/3 << " particles" << endl;
}

double get_ms(struct timespec t) {
        return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

void Update(void)
{
	solver->Update();
	step++;
	glutPostRedisplay();
}


//Rendering
void InitGL(void)
{
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(2);
	glMatrixMode(GL_PROJECTION);
}

//Rendering
void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 1);

	glColor4f(0.2f, 0.6f, 1.0f, 1);
	glBegin(GL_POINTS);
	for (auto& p : solver->particles)
		glVertex2f(p.x(0) * WINDOW_WIDTH, p.x(1) * WINDOW_HEIGHT);
	glEnd();

	glutSwapBuffers();
	if (step % INV_FPS == 0) {
		// save image output
		unsigned char* buffer = (unsigned char*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * 3);
		glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, buffer);
		string filepath = "./imgs/mpm" + to_string(frame) + ".png";
		stbi_flip_vertically_on_write(true);
		stbi_write_png(filepath.c_str(), WINDOW_WIDTH, WINDOW_HEIGHT, 3, buffer, WINDOW_WIDTH * 3);
		frame++;
		free(buffer);
	}

}

void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
	switch (c)
	{
	case ' ':
	case 'r':
	case 'R':
		InitMPM();
		break;
	}
}

int main(int argc, char** argv)
{
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("MPM");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);

	InitGL();
	InitMPM();

	glutMainLoop();
	return 0;
}
