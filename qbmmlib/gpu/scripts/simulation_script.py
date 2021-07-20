
import sys
import numpy as np

import time
from numba import njit, config, set_num_threads, threading_layer
from numpy.lib.type_check import nan_to_num

import pycuda.driver as cuda
import pycuda.autoinit

from qbmmlib.utils.jets_util import jet_initialize_moments

from pycuda.compiler import SourceModule

from chyqmom27_script import chyqmom27
from qbmmlib.utils.nquad import domain_get_fluxes, domain_invert_3d_rowmajor

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
        float *x, 
        float *y, 
        float *z, 
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

            // x, y, z  -> 2D array 
            //        idx in x direction (across)
            //        num_node in y-direction (down)
 
            // indices -> 2D array
            //         (x, y, z) in x_direction
            //         num_moments in y_direction 

            // f_min, f_max -> 3D array 
            //        idx in x direction (across) (num_point)
            //        num_moment in y-direction (down)
            //        num_node in z-direction (stack)
            
            for (int n = 0; n < num_nodes; n++) {
                for (int m = 0; m < num_moments; m++) {
                
                    float x_comp = x[m * num_points + idx];

                    float flux = wts[m * num_points + idx] * 
                            powf(x[m * num_points + idx],
                                indices[num_nodes * 3 + 0]);
                    flux *= powf(y[m * num_points + idx], 
                                indices[num_nodes * 3 + 1]);
                    flux *= powf(z[m * num_points + idx], 
                                indices[num_nodes * 3 + 2]);
                    
                    // printf("[%d] x = %f \\n", idx, x_comp);

                    float f1 = flux * fminf(x_comp, 0);
                    float f2 = flux * fmaxf(x_comp, 0);
    
                    fmax[n * num_points * num_moments + m * num_points + idx] = f1;
                    fmin[n * num_points * num_moments + m * num_points + idx] = f1;
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
# initialize jet 
num_coords = 3
num_nodes = 27
num_moments = 16
indices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
    ]
)

sizeof_float = np.int32(np.dtype(np.float32).itemsize)

## functions for setup
def projection(
        weights, abscissas, indices,
        num_coords, num_nodes):

    moments = np.zeros(indices.shape[0])
    ni = len(indices)
    for i in range(ni):
        if num_coords == 3:
            moments[i] = quadrature_3d(
                weights, abscissas, indices[i], num_nodes
            )
    return moments

def quadrature_3d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * abscissas[0, i] ** moment_index[0]
            * abscissas[1, i] ** moment_index[1]
            * abscissas[2, i] ** moment_index[2]
        )
    return q

def initialize(num_points):
    # states 
    ## Note: It is important to keep the parallelizable index (largest)
    ## on the most inner dimension
    state = cuda.aligned_zeros((num_moments, num_points), dtype=np.float32)

    wts_left, wts_right, xi_left, xi_right = jet_initialize_moments(num_coords, num_nodes)


    grid_spacing = 1/ (num_points - 2)
    disc_loc = 0.125
    n_pt = num_points - 2
    disc_idx = int(n_pt * disc_loc) - 2
    print('Dislocation index is ', disc_idx, ' out of ', n_pt, ' points')
    # print("abscissas left: ", xi_left[0,:])
    # print("abscissas right: ", xi_right[0,:])

    # Populate state
    moments_left = projection(wts_left, xi_left, indices,
            num_coords, num_nodes)
    moments_right = projection(wts_right, xi_right, indices,
            num_coords, num_nodes)

    state[:, :disc_idx] = np.asarray([moments_left]).T
    state[:, -disc_idx:] = np.asarray([moments_right]).T

    state[:, 0] = np.asarray([moments_right])
    state[:, -1] = np.asarray([moments_left])

    return state, grid_spacing

def single_advance_gpu(state, num_points, grid_space):

    rhs = cuda.aligned_zeros((num_moments, num_points), dtype=np.float32)
    time_before = cuda.Event()
    time_1 = cuda.Event()
    time_after = cuda.Event()
    ## allocate GPU memory 
    indices_device = cuda.mem_alloc_like(indices)
    cuda.memcpy_htod(indices_device, indices)

    f_min = cuda.mem_alloc(int(sizeof_float * 
                num_moments * num_nodes * num_points))
    f_max = cuda.mem_alloc(int(sizeof_float * 
                num_moments*num_nodes*num_points))
    
    flux_1 = cuda.mem_alloc_like(state)
    flux_2 = cuda.mem_alloc_like(state)

    ## compile GPU kernel 
    BlockSize = (256, 1, 1)
    GridSize = (num_points +BlockSize[0] - 1) /BlockSize[0];
    GridSize = (int(GridSize), 1, 1)

    domain_get_flux = QUAD.get_function('domain_get_flux_3d')
    fsum = QUAD.get_function('fsum_3d')
    flux_out = QUAD.get_function('flux_3d')
    ## compute_rhs 

    time_before.record()
    # grid_inversion(state)
    # output are pointer object to GPU memory 
    _, w, x, y, z = chyqmom27(state, num_points)

    time_1.record()

    # domain_get_fluxes(weights, abscissas, qbmm_mgr.indices,
    #                 num_points, qbmm_mgr.num_moments,
    #                 qbmm_mgr.num_nodes, flux)
    domain_get_flux(w, x, y, z, indices_device,
                    f_min, f_max, 
                    np.int32(num_moments), 
                    np.int32(num_nodes), 
                    np.int32(num_points),
                    block=BlockSize, grid=GridSize)

    fsum(flux_1, f_min, f_max, 
                    np.int32(num_moments), 
                    np.int32(num_nodes), 
                    np.int32(num_points),
                    block=BlockSize, grid=GridSize)
    flux_out(flux_1, flux_2, np.float32(grid_space), 
                    np.int32(num_moments), 
                    np.int32(num_points),
                    block=BlockSize, grid=GridSize)
    
    time_after.record()
    time_1.synchronize()
    time_after.synchronize()

    total_time = time_after.time_since(time_before)
    quad_time = time_after.time_since(time_1)
    
    cuda.memcpy_dtoh(rhs, flux_2)
    w.free()
    x.free()
    y.free()
    z.free()
    return rhs, total_time, quad_time

def single_advance_cpu(state, num_points, grid_spacing):
    weights = np.zeros([num_points, num_nodes])
    abscissas = np.zeros([num_points, num_coords, num_nodes])
    flux = np.zeros([num_points, num_moments])
    try: 
        time_before = time.perf_counter()
        domain_invert_3d_rowmajor(state, indices, 
                    weights, abscissas,
                    num_points, num_coords, num_nodes)
        time_1 = time.perf_counter()
        domain_get_fluxes(weights, abscissas, indices,
                    num_points, num_moments,
                    num_nodes, flux)
        rhs = flux / grid_spacing
        time_after= time.perf_counter()
        total_time = (time_after - time_before) * 1e3 # ms
        quad_time = (time_after - time_1) * 1e3 # ms
    except: 
        total_time = np.nan
        quad_time = np.nan
        rhs = None

    return rhs, total_time, quad_time
    

def time_script():
    res_file_name_quad = 'quad.csv' # output data file name
    res_file_name_total= 'total.csv' # output data file name
    max_input_size_mag = 6             # max number of input point (power of 10)
    num_points = 200                   # number of runs collected 
    trial = 5                          # For each run, the number of trials run. 
    num_device = 1                     # number of GPUs used

    config.THREADING_LAYER = 'threadsafe'
    set_num_threads(12)                # numba: number of concurrent CPU threads 
    print("Threading layer chosen: %s" % threading_layer())
    
    ## Header: 
    #  [num input, cpu_result (ms), gpu_result (ms)] 
    result_quad = np.zeros((num_points, 4))
    result_total= np.zeros((num_points, 4))

    this_result_cpu_total = np.zeros(trial)
    this_result_gpu_total = np.zeros(trial)
    this_result_cpu_quad = np.zeros(trial)
    this_result_gpu_quad = np.zeros(trial)

    # generate a set of input data size, linear in log space between 1 and maximum 
    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
    # for idx, in_size in enumerate(np.linspace(1e5, 1e6, num_points)):
        result_quad[idx, 0] = idx
        result_quad[idx, 1] = int(in_size)
        result_total[idx, 0] = idx
        result_total[idx, 1] = int(in_size)

        state, spacing = initialize(int(in_size))
        # output from GPU

        for i in range(0, trial, 1):
            # GPU time
            rhs, total_time, quad_time = single_advance_gpu(state, int(in_size), spacing)
            this_result_gpu_total[i] = total_time
            this_result_gpu_quad[i] = quad_time

            # numba time: 
            rhs, total_time, quad_time = single_advance_cpu(state, int(in_size), spacing)
            this_result_cpu_total[i] = total_time
            this_result_cpu_quad[i] = quad_time

        result_quad[idx, 2] = np.min(this_result_cpu_quad)
        result_quad[idx, 3] = np.min(this_result_gpu_quad)
        result_total[idx, 2] = np.min(this_result_cpu_total)
        result_total[idx, 3] = np.min(this_result_gpu_total)
    
        print("[{}/{}] running on {} inputs, CPU: {:4f}, GPU: {:4f}".format(
            idx, num_points, int(in_size), result_quad[idx, 2], result_quad[idx, 3]))


    np.savetxt(res_file_name_quad, result_quad, delimiter=',')
    np.savetxt(res_file_name_total, result_total, delimiter=',')

if __name__ == "__main__":
    time_script()
    
    # in_size = int(5e5)
    # state, space = initialize(in_size)
    # print(state.shape)
    # print(state[:, 5].shape)
    # single_advance_cpu(state, in_size, space)



