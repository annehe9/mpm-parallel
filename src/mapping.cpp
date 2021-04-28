// Keep mapping in separate file to prevent compilation issues.
#include <unordered_map>
#include <set>

#include "mapping.h"


std::unordered_map<int, std::set<int>> block_particles;
int grid_block_side;

// grid is arranged in blocks going left-to-right, up-to-down
int hash_particle(Particle p) {
        Vector2i base_coord = (p.x * INV_DX - Vector2d(0.5, 0.5)).cast<int>();
        // middle of 3x3 neighborhood, where base_coord is upper-left
        int gx = base_coord.x() + 1;
        int gy = base_coord.y() + 1;
        int bx = gx / grid_block_side;
        int by = gy / grid_block_side;
        return bx * grid_block_side + by;
}

void map_setup() {
        grid_block_side = (GRID_RES + BLOCKSIDE - 1) / BLOCKSIDE;
}

void map_particles(Particle *particles, int num_particles) {
        // reset sets
        for (int i = 0; grid_block_side * grid_block_side; i++) {
                block_particles[i].clear();
        }
        int hash_block;
        for (int i = 0; i < num_particles; i++) {
                hash_block = hash_particle(particles[i]);
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
