import pycuda
import pycuda.driver as cuda
from pycuda import compiler, gpuarray, tools
import threading
import numpy as np

import time
# -- initialize the device
import pycuda.autoinit

MATRIX_SIZE = 4000

# Inheritance for using thread
class GPUThread(threading.Thread):
    def __init__(self, number, arr):
        threading.Thread.__init__(self)
        self.number = number
        self.arr = arr

    def run(self):
        self.dev = cuda.Device(self.number)
        self.ctx = self.dev.make_context()

        # initialize gpu array and copy from cpu to gpu.
        self.array_gpu = gpuarray.to_gpu(self.arr)
         # Get lock to synchronize threads
        threadLock.acquire()

        ctic = time.time()
        np.dot(self.arr,self.arr)
        ctoc = float(time.time()-ctic)

        gtic = time.time()
        output = matmul(self.array_gpu,self.array_gpu)
        gtoc = float(time.time()-gtic)
        # Free lock to release next thread

        print("CPU:" , ctoc, "GPU:" , gtoc,"GPU-CPU:",gtoc-ctoc)
        threadLock.release()
        print("successful exit from thread %d \n" % self.number)
        self.ctx.pop()

        # delete device,context for saving resources.
        del self.ctx
        del self.array_gpu


def matmul(a_gpu,b_gpu,MATRIX_SIZE=MATRIX_SIZE):
    kernel_code_template = """
    __global__ void MatrixMulKernel(float *A, float *B, float *C)
    {

      const uint wA = %(MATRIX_SIZE)s;
      const uint wB = %(MATRIX_SIZE)s;

      // Block index
      const uint bx = blockIdx.x;
      const uint by = blockIdx.y;

      // Thread index
      const uint tx = threadIdx.x;
      const uint ty = threadIdx.y;

      // Index of the first sub-matrix of A processed by the block
      const uint aBegin = wA * %(BLOCK_SIZE)s * by;
      // Index of the last sub-matrix of A processed by the block
      const uint aEnd = aBegin + wA - 1;
      // Step size used to iterate through the sub-matrices of A
      const uint aStep = %(BLOCK_SIZE)s;

      // Index of the first sub-matrix of B processed by the block
      const uint bBegin = %(BLOCK_SIZE)s * bx;
      // Step size used to iterate through the sub-matrices of B
      const uint bStep = %(BLOCK_SIZE)s * wB;

      // The element of the block sub-matrix that is computed
      // by the thread
      float Csub = 0;
      // Loop over all the sub-matrices of A and B required to
      // compute the block sub-matrix
      for (int a = aBegin, b = bBegin;
           a <= aEnd;
           a += aStep, b += bStep)
        {
          // Shared memory for the sub-matrix of A
          __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
          // Shared memory for the sub-matrix of B
          __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

          // Load the matrices from global memory to shared memory
          // each thread loads one element of each matrix
          As[ty][tx] = A[a + wA * ty + tx];
          Bs[ty][tx] = B[b + wB * ty + tx];
          // Synchronize to make sure the matrices are loaded
          __syncthreads();

          // Multiply the two matrices together;
          // each thread computes one element
          // of the block sub-matrix
          for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
            Csub += As[ty][k] * Bs[k][tx];

          // Synchronize to make sure that the preceding
          // computation is done before loading two new
          // sub-matrices of A and B in the next iteration
          __syncthreads();
        }

      // Write the block sub-matrix to global memory;
      // each thread writes one element
      const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
      C[c + wB * ty + tx] = Csub;
    }
    """

    # define size of blocks and tiles sub-matrix
    # (we assume that the block size is same as tile size)
    TILE_SIZE = 20
    BLOCK_SIZE = TILE_SIZE

    # get the kernel code from the template
    # by specifying the constants MATRIX_SIZE and BLOCK_SIZE
    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'BLOCK_SIZE': BLOCK_SIZE,
        }

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")

    # call the kernel on the card
    matrixmul(
        # inputs
        a_gpu, b_gpu,
        # output
        c_gpu,
        # grid of multiple blocks
        grid = (MATRIX_SIZE // TILE_SIZE, MATRIX_SIZE // TILE_SIZE),
        # block of multiple threads
        block = (TILE_SIZE, TILE_SIZE, 1),
        )

    return c_gpu


num = cuda.Device.count()
gpu_thread_list = []

some_list = []
for i in range(num):
    some_list.append(np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32))
a_gpu = gpuarray.to_gpu(some_list[0])
b_gpu = gpuarray.to_gpu(some_list[0])

a = matmul(a_gpu,b_gpu)
print("MATRIX SIZE: ", MATRIX_SIZE)
print("Difference between GPU result and CPU result")
print(np.dot(some_list[0],some_list[0])-a.get())

threadLock = threading.Lock()
for i,arr in enumerate(some_list):
    gpu_thread = GPUThread(i,arr)
    gpu_thread.start()
    gpu_thread_list.append(gpu_thread)