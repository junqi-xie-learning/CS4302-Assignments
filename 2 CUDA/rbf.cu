/* File:
 *     rbf.cu
 *
 *
 * Idea:
 *     Computes a parallel rbf kernel of matrices with CUDA.  Each thread will
 *     be responsible for calculating one element of the RBF kernel matrix.
 *
 * Compile:
 *     nvcc -o rbf.out rbf.cu
 * Usage:
 *     ./rbf.out <sigma> <dimension> <m> <n>
 *
 * Input:
 *     None unless compiled with debug mode.
 *     If in debug mode, read matrix `A`, `B` from standard input.
 * Output:
 *     Elapsed time for the computation
 *     If in debug mode, print the RBF kernel matrix.
 */

#include <iostream>
#include <random>
#include "cuda_runtime.h"
using namespace std;

bool debug = false;

/*------------------------------------------------------------------
 * Function:  rbf_kernel
 * Purpose:   Kernel function to compute an element in the matrix produced by
 *            RBF kernels of row vectors of `A` and `B`.
 *            Note that `A_d`, `B_d`, and `C_d` are in the device memory.
 * In args:   A_d, B_d, sigma, dimension, m, n
 * Out arg:   C_d[row * n + col]
 */
template <typename T>
__global__ void rbf_kernel(T A_d[], T B_d[], T C_d[], T sigma, int dimension, int m, int n)
{
    // Calculate the row index of the working element in C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the working element in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < m) && (col < n))
    {
        T dist{ };
        // Each thread computes one element of the block sub-matrix
        for (int i = 0; i < dimension; ++i)
        {
            T diff = A_d[row * dimension + i] - B_d[col * dimension + i];
            dist += diff * diff;
        }
        C_d[row * n + col] = exp(-dist / (2 * sigma * sigma));
    }
}

/*------------------------------------------------------------------
 * Function:  rbf
 * Purpose:   Wrapper function to compute the RBF kernel of `A` and `B`.
 *            where features are row vectors.
 * In args:   A, B, sigma, dimension, m, n
 * Out arg:   C
 */
template <typename T>
void rbf(T A[], T B[], T C[], T sigma, int dimension, int m, int n)
{
    int A_size = m * dimension * sizeof(T), B_size = n * dimension * sizeof(T),
        C_size = m * n * sizeof(T);
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
    dim3 dimBlock(m, n);
    rbf_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, sigma, dimension, m, n);

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
    double sigma = stod(argv[1]);
    int dimension = stoi(argv[2]), m = stoi(argv[3]), n = stoi(argv[4]);

    // Generate matrices
    double *A = nullptr, *B = nullptr;
    if (debug)
    {
        cout << "Enter matrix A: " << endl;
        A = read_matrix(dimension, m);
        cout << "Enter matrix B: " << endl;
        B = read_matrix(dimension, n);
    }
    else
    {
        cout << "Generated matrix A of size " << dimension << " * " << m << ", "
             << "matrix B of size " << dimension << " * " << n << endl;
        A = generate_matrix(dimension, m);
        B = generate_matrix(dimension, n);
    }

    // Initialize CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    // Call `rbf` and get the time elapsed
    double *C = new double[m * n];
    cudaEventRecord(start);

    rbf(A, B, C, sigma, dimension, m, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "RBF kernel calculated. Elapsed time: " << elapsed << " seconds" << endl;

    if (debug)
    {
        cout << "The RBF kernel is: " << endl;
        print_matrix(C, m, n);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
