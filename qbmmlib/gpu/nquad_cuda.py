from pycuda.compiler import SourceModule

QUAD = SourceModule('''
    // set a segment of memory to a specific value
    __global__ void float_value_set(float *addr, float value, int size, int offset) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            addr[idx + offset] = value;
            //printf("[%d], setting addr to %f \\n", idx, value);
        }
    }

    // set a segment of memory to a specific array
    __global__ void float_array_set(float *addr, float *value, int size, int loc_d, int loc_s) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {

            addr[idx + loc_d] = value[idx + loc_s];
            //printf("[%d]loc_s: %d, setting addr[%d] to %f \\n", idx, loc_s, loc_d+idx, value[idx]);
        }
    }

    __global__ void domain_get_flux_3d(
        float *wts, 
        float *xi, 
        float *indices,
        float *fmin,
        float *fmax,
        int num_moments, 
        int num_nodes, 
        int num_points)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < num_points; idx+=blockDim.x*gridDim.x) {
            
            // wts -> 2D array, idx in x-direction (across), num_node in y-direction (down)

            // xi  -> 3D array 
            //        idx in x direction (across)
            //        num_node in y-direction (down)
            //        (x, y, z) in z-direction (stack)

            // indices -> 2D array
            //         (x, y, z) in x_direction
            //         num_moments in y_direction 

            // f_min, f_max -> 3D array 
            //        idx in x direction (across)
            //        num_moment in y-direction (down)
            //        num_node in z-direction (stack)
            
            for (int m = 0; m < num_moments; m++) {
                for (int n = 0; n < num_nodes; n++) {
                    float x_comp = xi[0 * num_points * num_nodes + n * num_points + idx];

                    float flux = wts[n * num_points + idx] * 
                            powf(xi[0 * num_points * num_nodes + n * num_points + idx],
                                indices[num_nodes * 3 + 0]);
                    flux *= powf(xi[1 * num_points * num_nodes + n * num_points + idx], 
                                indices[num_nodes * 3 + 1]);
                    flux *= powf(xi[2 * num_points * num_nodes + n * num_points + idx], 
                                indices[num_nodes * 3 + 2]);

                
                    fmin[n * num_points * num_moments + m * num_points + idx] = flux * fminf(x_comp, 0);
                    fmin[n * num_points * num_moments + m * num_points + idx] = flux * fmaxf(x_comp, 0);
                }
            }
        }
    }

    __global__ void fsum_3d(
        float *flux, 
        float *f_min, 
        float *f_max,
        int num_moments, 
        int num_nodes, 
        int num_points)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx + 1; idx < num_points - 1; idx+=blockDim.x*gridDim.x) {
            
            for (int m = 0; m < num_moments; m++) {

                float sum = 0;
                for (int n = 0; n < num_nodes; n++) {

                    sum += f_max[n * num_points * num_moments + m * num_points + idx -1];
                    sum += f_min[n * num_points * num_moments + m * num_points + idx];

                }
                flux[m * num_points + idx] = sum;
            }

        }
    }

    __global__ void flux_3d(
        float* flux_old, 
        float* flux_new, 
        float grid_space,
        int num_moments, 
        int num_points)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx + 1; idx < num_points - 2; idx+=blockDim.x*gridDim.x) {
            for(int m = 0; m < num_moments; m++) {
                flux_new[m * num_points + idx] = 
                    flux_old[m * num_points + idx] + flux_old[m * num_points + idx + 1];
                
                flux_new[m * num_points + idx] /= grid_space;
            }
        }
    }
    
''')

