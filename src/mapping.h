#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

#include "particle.h"

void map_particles(Particle *particles, int num_particles);
int retrieve_block(int index, Particle *particles, Particle *particle_set);
