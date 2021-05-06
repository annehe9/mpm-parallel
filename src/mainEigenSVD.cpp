#if __APPLE__
#include <GLUT/glut.h>
#else
//#include <windows.h>
#include <GL/glut.h>
#endif

#include <time.h>
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//#include <pthread.h>
//#include <omp.h>
//#include "helper.h"

//int NCORES = omp_get_num_procs();
//int MAX_NTHREADS = omp_get_max_threads();

// References:
// https://github.com/yuanming-hu/taichi_mpm/blob/master/mls-mpm88-explained.cpp
// https://lucasschuermann.com/writing/implementing-sph-in-2d for visualization

// Particle representation
struct Particle
{
    Particle(double _x, double _y) : x(_x, _y), v(0.0, 0.0), F(Matrix2d::Identity()), C(Matrix2d::Zero()), Jp(1.0) {}
    Vector2d x, v; //position and velocity
    Matrix2d F, C; //deformation gradient, APIC momentum
    double Jp; //determinant of deformation gradient, which is volume
};

// Granularity
const static int MAX_PARTICLES = 2500;
const static int BLOCK_PARTICLES = 500;		// number of particles added in a block
int NUM_PARTICLES = 0;					// keeps track of current number of particles
const static int GRID_RES = 80;				// grid dim of one side
const static int NUM_CELLS = GRID_RES * GRID_RES;	// number of cells in the grid
const static double DT = 0.00001;			// integration timestep
const static double DX = 1.0 / GRID_RES;
const static double INV_DX = 1.0 / DX;

// Data structures
static vector<Particle> particles;
// Vector3: [velocity_x, velocity_y, mass]
static Vector3d grid[GRID_RES][GRID_RES];
//static Cell grid[GRID_RES][GRID_RES];
static size_t grididxs[GRID_RES * GRID_RES];
static size_t num_filled = 0;

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
const static double VIEW_WIDTH = 800;
const static double VIEW_HEIGHT = 600;

int iterations = 20000;
double totalTime = 0;
double avgP2G = 0;
double avgGrid = 0;
double avgG2P = 0;

// yay add more particles randomly in a square
void addParticles(double xcenter, double ycenter)
{
    for (int i = 0; i < BLOCK_PARTICLES; i++) {
        particles.push_back(Particle((((double)rand() / (double)RAND_MAX) * 2 - 1) * 0.08 + xcenter, (((double)rand() / (double)RAND_MAX) * 2 - 1) * 0.08 + ycenter));
    }
    NUM_PARTICLES += BLOCK_PARTICLES;
}

void InitMPM(void)
{
    cout << "initializing mpm with " << BLOCK_PARTICLES << " particles" << endl;
    addParticles(0.55, 0.45);
    addParticles(0.45, 0.65);
    addParticles(0.55, 0.85);
}

void P2G_iteration(Particle& p) {

    Matrix2d PF_0, PF_1, PF, stress, affine;
    Vector3d mass_x_velocity;
    Vector2i base_coord;
    Vector2d fx, tmpa, tmpb, tmpc, dpos, tmp, w[3];
    Vector2d onehalf(1.5, 1.5); //have to make these so eigen doesn't give me errors
    Vector2d one(1.0, 1.0);
    Vector2d half(0.5, 0.5);
    Vector2d threequarters(0.75, 0.75);
    double e, mu, lambda, J, Dinv, pf1tmp;
    int i, j;

    base_coord = (p.x * INV_DX - half).cast<int>();
    fx = p.x * INV_DX - base_coord.cast<double>();

    // Quadratic kernels [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 123, with x=fx, fx-1,fx-2]
    tmpa = onehalf - fx;
    tmpb = fx - one;
    tmpc = fx - half;
    w[0] = 0.5 * (tmpa.cwiseProduct(tmpa));
    w[1] = threequarters - (tmpb.cwiseProduct(tmpb));
    w[2] = 0.5 * (tmpc.cwiseProduct(tmpc));

    // Snow
    // Compute current Lamé parameters [https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf Eqn. 87]
    e = exp(HARD * (1.0 - p.Jp));
    mu = MU_0 * e;
    lambda = LAMBDA_0 * e;

    // Current volume
    J = p.F.determinant();

    // Polar decomposition for fixed corotated model, https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf paragraph after Eqn. 45
    JacobiSVD<Matrix2d> svd(p.F, ComputeFullU | ComputeFullV);
    Matrix2d r = svd.matrixU() * svd.matrixV().transpose();

    // [http://mpm.graphics Paragraph after Eqn. 176]
    Dinv = 4 * INV_DX * INV_DX;
    // [http://mpm.graphics Eqn. 52]
    PF_0 = (2 * mu) * (p.F - r) * p.F.transpose();
    pf1tmp = (lambda * (J - 1) * J);
    PF_1 << pf1tmp, pf1tmp,
        pf1tmp, pf1tmp;
    PF = PF_0 + PF_1;

    // Cauchy stress times dt and inv_dx
    stress = -(DT * VOL) * (Dinv * PF);

    // Fused APIC momentum + MLS-MPM stress contribution
    // See https://yzhu.io/publication/mpmmls2018siggraph/paper.pdf
    // Eqn 29
    affine = stress + MASS * p.C;

    // P2G
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            dpos = (Vector2d(i, j) - fx) * DX;
            // Translational momentum
            mass_x_velocity = Vector3d(p.v.x() * MASS, p.v.y() * MASS, MASS);
            tmp = affine * dpos;
            grid[base_coord.x() + i][base_coord.y() + j] += (
                w[i].x() * w[j].y() * (mass_x_velocity + Vector3d(tmp.x(), tmp.y(), 0))
                );
        }
    }
}

void P2G(void)
{
    memset(grid, 0, sizeof(grid));

    size_t i;
    //#pragma omp parallel for schedule(dynamic) private(i)
    for (i = 0; i < particles.size(); i++) {//(Particle &p : particles) {
        P2G_iteration(particles[i]);
    }
}

void CollectGridIndices(void) {
    size_t i, j;
    num_filled = 0;
    for (i = 0; i < GRID_RES; i++) {
        for (j = 0; j < GRID_RES; j++) {
            Vector3d& g = grid[i][j];
            if (g[2] > 0) {
                grididxs[num_filled] = (i * GRID_RES + j);
                num_filled++;
            }
        }
    }
}

void UpdateGridVelocity_iteration(Vector3d& g, int i, int j) {
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

void UpdateGridVelocity(void) {
    size_t i;
    // TODO: Does not appear omp is useful here, but if must use, static
    // is better than dynamic scheduling.
    // Consider that only some of the iterations actually have work.
    // Most others are just a check.
//#pragma omp parallel for schedule(static) private(i)
    for (i = 0; i < num_filled; i++) {
        size_t idx = grididxs[i];
        Vector3d& g = grid[idx / GRID_RES][idx % GRID_RES];
        UpdateGridVelocity_iteration(g, idx / GRID_RES, idx % GRID_RES);
        //cout << "grid vel:" << g[1] << endl;
    }
}

void G2P_iteration(Particle& p) {
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
            Vector3d curr = grid[base_coord.x() + i][base_coord.y() + j];
            Vector2d grid_v(curr.x(), curr.y());
            //cout << "grid vel:" << grid_v.y() << endl;
            double weight = w[i].x() * w[j].y();
            //cout << "weight: " << weight << endl;
            // Velocity
            p.v += weight * grid_v;
            // APIC C, outer product of weighted velocity and dist, paper equation 10
            p.C += 4 * INV_DX * ((weight * grid_v) * dpos.transpose());
        }
    }

    // Advection
    //double tempy = p.x.y();
    p.x += DT * p.v;
    /*
    if (tempy - p.x.y() > 0) {
        cout << "change" << tempy - p.x.y() << endl;
    }
    cout << "velocity " << p.v << endl;
    */

    // MLS-MPM F-update eqn 17
    Matrix2d F = (Matrix2d::Identity() + DT * p.C) * p.F;

    JacobiSVD<Matrix2d> svd(F, ComputeFullU | ComputeFullV);
    Matrix2d svd_u = svd.matrixU();
    Matrix2d svd_v = svd.matrixV();
    // Snow Plasticity
    Vector2d sigvalues = svd.singularValues().array().min(1.0f + 7.5e-3).max(1.0 - 2.5e-2);
    Matrix2d sig = sigvalues.asDiagonal();

    double oldJ = F.determinant();
    F = svd_u * sig * svd_v.transpose();

    double Jp_new = min(max(p.Jp * oldJ / F.determinant(), 0.6), 20.0);

    p.Jp = Jp_new;
    p.F = F;

}

void G2P(void)
{
    size_t i;
    //#pragma omp parallel for schedule(static) private(i)
    for (i = 0; i < particles.size(); i++) {//(Particle &p : particles) {
        G2P_iteration(particles[i]);
    }
}

double get_ms(struct timespec t) {
    return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

void Update(void)
{
    //struct timespec t1, t2, t3, t4;
    //clock_gettime(CLOCK_REALTIME, &t1);
    P2G();
    //clock_gettime(CLOCK_REALTIME, &t2);
    CollectGridIndices();
    UpdateGridVelocity();
    //clock_gettime(CLOCK_REALTIME, &t3);
    G2P();
    //clock_gettime(CLOCK_REALTIME, &t4);
    /*
    printf("Total: %.3f\nP2G: %.3f\tUpdate: %.3f\tG2P: %.3f\n",
            get_ms(t4) - get_ms(t1),
            get_ms(t2) - get_ms(t1),
            get_ms(t3) - get_ms(t2),
            get_ms(t4) - get_ms(t3)
    );
    */
    /*
    totalTime += get_ms(t4) - get_ms(t1);
    avgP2G += get_ms(t2) - get_ms(t1);
    avgGrid += get_ms(t3) - get_ms(t2);
    avgG2P += get_ms(t4) - get_ms(t3);
    */
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
    glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    for (int i = 0; i < particles.size(); i++) {
        Particle& p = particles[i];
        glVertex2f(p.x(0) * VIEW_WIDTH, p.x(1) * VIEW_HEIGHT);
    }
    glEnd();

    glutSwapBuffers();
}

void SaveFrame(int frame)
{
    // save image output
    unsigned char* buffer = (unsigned char*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * 3);
    glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, buffer);
    string filepath = "./imgs/mpm" + to_string((long long int)frame) + ".png";
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg(filepath.c_str(), WINDOW_WIDTH, WINDOW_HEIGHT, 3, buffer, 100);
    free(buffer);
}

void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
    switch (c)
    {
    case ' ':
        if (particles.size() >= MAX_PARTICLES)
            std::cout << "maximum number of particles reached" << std::endl;
        else
        {
            addParticles(0.5, 0.5);
        }
        break;
    case 'r':
    case 'R':
        particles.clear();
        InitMPM();
        break;
    }
}

int main(int argc, char** argv)
{
    /*
    printf("Original cores: %d\tOriginal max threads: %d\n",
        NCORES, MAX_NTHREADS
    );
    // OpenMP
    if (argc < 2) {
        printf("Input # of cores.");
        exit(-1);
    }
    NCORES = atoi(argv[1]);
    printf("NCORES: %d\tMAX_NTHREADS: %d\n", NCORES, MAX_NTHREADS);
    omp_set_num_threads(NCORES);
    // Eigen
    Eigen::initParallel();
    Eigen::setNbThreads(NCORES);
    */

    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInit(&argc, argv);
    glutCreateWindow("MPM");
    //glutDisplayFunc(Render);
    //glutIdleFunc(Update);
    //glutKeyboardFunc(Keyboard);

    InitGL();
    InitMPM();
    for (int i = 0; i < iterations; i++) {
        Update();
        if (i % 10 == 0) {
            Render();
            SaveFrame(i / 10);
        }
        printf("Iteration: %d\n", i);
    }
    //printf("Benchmark over %d iterations\n", iterations);
    //printf("Total time: %.3f\nAverage time per iteration: %.3f\nAverage time for P2G: %.3f\nAverage time for grid update: %.3f\nAverage time for G2P: %.3f\n",
        //totalTime, totalTime / iterations, avgP2G / iterations, avgGrid / iterations, avgG2P / iterations);
    //glutMainLoop();
    return 0;
}
