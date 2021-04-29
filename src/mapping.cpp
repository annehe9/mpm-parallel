// Keep mapping in separate file to prevent compilation issues.
#include <stdio.h>
#include <iostream>
using namespace std;

#include <unordered_map>
#include <set>

#include "mapping.h"


std::unordered_map<int, std::set<int>> block_particles;

// grid is arranged in blocks going left-to-right, up-to-down
int hash_particle(Particle p) {
        Vector2i base_coord = (p.x * INV_DX - Vector2d(0.5, 0.5)).cast<int>();
        // middle of 3x3 neighborhood, where base_coord is upper-left
        int gx = base_coord.x() + 1;
        int gy = base_coord.y() + 1;
        int bx = gx / OFFSIDE;// GRID_BLOCK_SIDE;
        int by = gy / OFFSIDE;//GRID_BLOCK_SIDE;
        if (DEBUG) printf("(%f, %f) => (%d, %d) => (%d, %d)\n", p.x(0), p.x(1), gx, gy, bx, by);
        return bx * GRID_BLOCK_SIDE + by;
}

void map_particles(Particle *particles, int num_particles) {
        // reset sets
        for (int i = 0; i < GRID_BLOCK_SIDE * GRID_BLOCK_SIDE; i++) {
                block_particles[i].clear();
        }
        int hash_block;
        for (int i = 0; i < num_particles; i++) {
                hash_block = hash_particle(particles[i]);
                if (DEBUG) cout << "map: " << i << ", " << hash_block << endl;
                block_particles[hash_block].insert(i);     
        }        
}

// change particle_set, return new size
int retrieve_block(int index, Particle *particles, Particle *particle_set) {
        set<int> pset = block_particles[index];
        int count = 0;
        for (int x : pset) {
               particle_set[count] = particles[x];
               count += 1; 
        }
        return count;
}
