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

// Image output params
static int frame = 0;	// current image
static int step = 0;	// current simulation step
const static int FPS = 500; // fps of output video
const static int INV_FPS = (1.0 / DT) / FPS;

cudaMPM *solver;

void InitMPM(void)
{
	solver = new cudaMPM();
	solver->setup();
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

        for (int i = 0; i < solver->NUM_PARTICLES; i++) {
                Particle p = solver->particles[i];
        	glVertex2f(p.x(0) * WINDOW_WIDTH, p.x(1) * WINDOW_HEIGHT);
        }
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
        /*
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("MPM");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);

	InitGL();
	InitMPM();

	glutMainLoop();
        */
	solver = new cudaMPM();
	solver->setup();
        int ITERATIONS = 20000;
        for (int i=0; i < ITERATIONS; i++) {
                solver->Update();
        }

	return 0;
}
