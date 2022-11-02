/* File:
 *     mat_mult.cu
 *
 *
 * Idea:
 *     Computes a parallel product of matrices with CUDA.  If not `tiled`, each
 *     thread will be responsible for calculating one element of the product 
 *     matrix.  If `tiled`, each thread will be responsible for loading part of
 *     the matrices into shared memory and then calculating part of the product
 *     matrix.
 *     Note that the TILE_WIDTH should be tuned according to the workload and
 *     the GPU architecture to achieve the best performance.
 *
 * Compile:
 *     nvcc -o mat_mult.out mat_mult.cu
 * Usage:
 *     ./mat_mult.out <m> <n> <k>
 *
 * Input:
 *     None unless compiled with debug mode.
 *     If in debug mode, read matrix `A`, `B` from standard input.
 * Output:
 *     Elapsed time for the computation
 *     If in debug mode, print the product of the matrices.
 */

#include <iostream>
#include <random>
#include "cuda_runtime.h"
using namespace std;

bool debug = false;
bool tiled = true;
const int TILE_WIDTH = 16;

/*------------------------------------------------------------------
 * Function:  mat_mult_kernel
 * Purpose:   Kernel function to compute an element in the matrix multiplied by
 *            matrix `A` of size `m` * `n` and `B` of size `n` * `k`.
 *            Note that `A_d`, `B_d`, and `C_d` are in the device memory.
 * In args:   A_d, B_d, m, n, k
 * Out arg:   C_d[row * k + col]
 */
template <typename T>
__global__ void mat_mult_kernel(T A_d[], T B_d[], T C_d[], int m, int n, int k)
{
    // Calculate the row index of the working element in C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the working element in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < m) && (col < k))
    {
        T value{ };
        // Each thread computes one element of the block sub-matrix
        for (int i = 0; i < n; ++i)
        {
            value += A_d[row * n + i] * B_d[i * k + col];
        }
        C_d[row * k + col] = value;
    }
}

/*------------------------------------------------------------------
 * Function:  mat_mult
 * Purpose:   Wrapper function to multiply matrix `A` of size `m` * `n` and
 *            `B` of size `n` * `k`.
 * In args:   A, B, m, n, k
 * Out arg:   C
 */
template <typename T>
void mat_mult(T A[], T B[], T C[], int m, int n, int k)
{
    int A_size = m * n * sizeof(T), B_size = n * k * sizeof(T),
        C_size = m * k * sizeof(T);
    T *A_d, *B_d, *C_d;

    // Transfer A and B to device memory
    cudaMalloc((void **)&A_d, A_size);
    cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&B_d, B_size);
    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    cudaMalloc((void **)&C_d, C_size);

    // Kernel Invocation
    dim3 dimGrid(1, 1);
    dim3 dimBlock(m, k);
    mat_mult_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, k);

    // Transfer C from device to host
    cudaMemcpy(C, C_d, C_size, cudaMemcpyDeviceToHost);
    // Free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/*------------------------------------------------------------------
 * Function:  mat_mult_tiled_kernel
 * Purpose:   Kernel function to compute an element in the matrix multiplication
 *            with optimization of tiling and utilization of shared memory.
 *            Note that `A_d`, `B_d`, and `C_d` are in the device memory.
 * In args:   A_d, B_d, m, n, k
 * Out arg:   C_d[row * k + col]
 */
template <typename T>
__global__ void mat_mult_tiled_kernel(T A_d[], T B_d[], T C_d[], int m, int n, int k)
{
    __shared__ T A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ T B_shared[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the working element in C
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Loop over the A and B tiles required to compute P element
    T value{ };
    for (int i = 0; i < n; i += TILE_WIDTH)
    {
        // Collaborative loading of M and N tiles into shared memory
        if ((row < m) && (i + tx) < n)
        {
            A_shared[ty][tx] = A_d[row * n + i + tx];
        }
        if ((i + ty) < n && (col < k))
        {
            B_shared[ty][tx] = B_d[(i + ty) * k + col];
        }
        __syncthreads();

        for (int ii = 0; ii < TILE_WIDTH; ++ii)
        {
            value += A_shared[ty][ii] * B_shared[ii][tx];
        }
        __syncthreads();
    }

    if ((row < m) && (col < k))
    {
        C_d[row * k + col] = value;
    }
}

/*------------------------------------------------------------------
 * Function:  mat_mult_tiled
 * Purpose:   Wrapper function of matrix multiplication with optimization of
 *            tiling and utilization of shared memory.
 * In args:   A, B, m, n, k
 * Out arg:   C
 */
template <typename T>
void mat_mult_tiled(T A[], T B[], T C[], int m, int n, int k)
{
    int A_size = m * n * sizeof(T), B_size = n * k * sizeof(T),
        C_size = m * k * sizeof(T);
    T *A_d, *B_d, *C_d;

    // Transfer A and B to device memory
    cudaMalloc((void **)&A_d, A_size);
    cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&B_d, B_size);
    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    cudaMalloc((void **)&C_d, C_size);

    // Kernel Invocation
    dim3 dimGrid(TILE_WIDTH, TILE_WIDTH);
    dim3 dimBlock(ceil(m / (double)TILE_WIDTH), ceil(k / (double)TILE_WIDTH));
    mat_mult_tiled_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, k);

    // Transfer C from device to host
    cudaMemcpy(C, C_d, C_size, cudaMemcpyDeviceToHost);
    // Free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/*------------------------------------------------------------------
 * Function: generate_matrix
 * Purpose:  Use the random number generator random to generate
 *           the entries in `A` in [0, 0]
 * In arg:   m, n
 * Out arg:  A
 */
double *generate_matrix(int m, int n)
{
    default_random_engine generator;
    uniform_real_distribution<double> distribution{0, 0};

    double *A = new double[m * n];
    for (int i = 0; i < m * n; i++)
        A[i] = distribution(generator);
    return A;
}

/*------------------------------------------------------------------
 * Function: read_matrix
 * Purpose:  Read in a matrix
 * In arg:   m, n
 * Out arg:  A
 */
double *read_matrix(int m, int n)
{
    double *A = new double[m * n];
    for (int i = 0; i < m * n; i++)
        cin >> A[i];
    return A;
}

/*------------------------------------------------------------------
 * Function: print_matrix
 * Purpose:  Print a matrix
 * In args:  A, m, n
 */
void print_matrix(double *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i * n + j] << " ";
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    // Get command line args
    int m = stoi(argv[1]), n = stoi(argv[2]), k = stoi(argv[3]);

    // Generate matrices
    double *A = nullptr, *B = nullptr;
    if (debug)
    {
        cout << "Enter matrix A: " << endl;
        A = read_matrix(m, n);
        cout << "Enter matrix B: " << endl;
        B = read_matrix(n, k);
    }
    else
    {
        cout << "Generated matrix A of size " << m << " * " << n << ", "
             << "matrix B of size " << n << " * " << k << endl;
        A = generate_matrix(m, n);
        B = generate_matrix(n, k);
    }

    // Initialize CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    // Call `mat_mult` and get the time elapsed
    double *C = new double[m * k];
    cudaEventRecord(start);

    if (tiled)
    {
        mat_mult_tiled(A, B, C, m, n, k);
    }
    else
    {
        mat_mult(A, B, C, m, n, k);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "Product calculated. Elapsed time: " << elapsed << " seconds" << endl;

    if (debug)
    {
        cout << "The product is: " << endl;
        print_matrix(C, m, k);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}