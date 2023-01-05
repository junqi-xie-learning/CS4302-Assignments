/* File:
 *     mat_mult.cu
 *
 *
 * Idea:
 *     Computes a parallel product of matrices with Tensor Core.  The computation
 *     leverages Tensor Cores to accelerate matrix problems of the form D=A*B+C.
 *     These operations are supported on mixed-precision floating point data for
 *     devices of compute capability 7.0 or higher.  The operations are organized
 *     in warp-level matrix multiply operations, with a tile of 16x16 elements.
 *
 * Compile:
 *     nvcc -arch=sm_70 -o mat_mult.out mat_mult.cu
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
#include "mma.h"
using namespace std;
using namespace nvcuda;

bool debug = false;
const int TILE_WIDTH = 16;
const int WARP_SIZE = 32;

/*------------------------------------------------------------------
 * Function:  wmma_kernel
 * Purpose:   Kernel function to compute a tile of elements in the matrix
 *            multiplication with warp-level matrix multiplication.
 *            Note that `A_d`, `B_d`, and `C_d` are in the device memory,
 *            and `m`, `n`, `k` are indices of the tile.
 * In args:   A_d, B_d, m, n, k
 * Out arg:   C_d[row * k + col]
 */
__global__ void wmma_kernel(half A_d[], half B_d[], float C_d[], int m, int n, int k)
{
    // Calculate the row index of the working element in C
    int row = blockIdx.z * blockDim.z + threadIdx.z;
    // Calculate the column index of the working element in C
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < m) && (col < k))
    {
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::col_major> B_frag;
        wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, float> C_frag;

        // Initialize the output to zero
        wmma::fill_fragment(C_frag, 0.0f);

        // Each thread computes one tile of the block sub-matrix
        for (int i = 0; i < n; ++i)
        {
            // Load the inputs
            wmma::load_matrix_sync(A_frag, &A_d[row * n * TILE_WIDTH * TILE_WIDTH + i * TILE_WIDTH], n * TILE_WIDTH);
            wmma::load_matrix_sync(B_frag, &B_d[col * n * TILE_WIDTH * TILE_WIDTH + i * TILE_WIDTH], n * TILE_WIDTH);

            // Perform the matrix multiplication
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }

        // Store the output
        wmma::store_matrix_sync(&C_d[row * k * TILE_WIDTH * TILE_WIDTH + col * TILE_WIDTH], C_frag, k * TILE_WIDTH, wmma::mem_row_major);
    }
}

/*------------------------------------------------------------------
 * Function:  wmma_mat_mult
 * Purpose:   Wrapper function of matrix multiplication with warp-level matrix
 *            multiplication.
 *            Note that matrix `A` is of size `m` * `n` and `B` is of size `n` * `k`.
 * In args:   A, B, m, n, k
 * Out arg:   C
 */
void wmma_mat_mult(half A[], half B[], float C[], int m, int n, int k)
{
    int m_tile = ceil(m / (float)TILE_WIDTH), n_tile = ceil(n / (float)TILE_WIDTH),
        k_tile = ceil(k / (float)TILE_WIDTH);

    int A_size = m_tile * n_tile * TILE_WIDTH * TILE_WIDTH * sizeof(half),
        B_size = k_tile * n_tile * TILE_WIDTH * TILE_WIDTH * sizeof(half),
        C_size = m_tile * k_tile * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    half *A_d, *B_d;
    float *C_d;

    // Transfer A and B to device memory
    cudaMalloc((void **)&A_d, A_size);
    cudaMemset(A_d, 0, A_size);
    for (int i = 0; i < m; ++i)
    {
        cudaMemcpy(&A_d[i * n_tile * TILE_WIDTH], &A[i * n], n * sizeof(half), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void **)&B_d, B_size);
    cudaMemset(B_d, 0, B_size);
    for (int i = 0; i < k; ++i)
    {
        cudaMemcpy(&B_d[i * n_tile * TILE_WIDTH], &B[i * n], n * sizeof(half), cudaMemcpyHostToDevice);
    }

    // Allocate C in device memory
    cudaMalloc((void **)&C_d, C_size);

    // Kernel Invocation
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(WARP_SIZE, m_tile, k_tile);
    wmma_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m_tile, n_tile, k_tile);

    // Transfer C from device to host
    for (int i = 0; i < m; ++i)
    {
        cudaMemcpy(&C[i * k], &C_d[i * k_tile * TILE_WIDTH], k * sizeof(float), cudaMemcpyDeviceToHost);
    }
    // Free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/*------------------------------------------------------------------
 * Function: generate_matrix
 * Purpose:  Use the random number generator random to generate
 *           the entries in `A` in [0.0, 1.0]
 * In arg:   m, n
 * Out arg:  A
 */
half *generate_matrix(int m, int n)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution{ 0.0, 1.0 };

    half *A = new half[m * n];
    for (int i = 0; i < m * n; i++)
    {
        float A_i = distribution(generator);
        A[i] = __float2half(A_i);
    }
    return A;
}

/*------------------------------------------------------------------
 * Function: read_matrix
 * Purpose:  Read in a matrix
 * In arg:   m, n
 * Out arg:  A
 */
half *read_matrix(int m, int n)
{
    half *A = new half[m * n];
    for (int i = 0; i < m * n; i++)
    {
        float A_i = 0;
        cin >> A_i;
        A[i] = __float2half(A_i);
    }
    return A;
}

/*------------------------------------------------------------------
 * Function: print_matrix
 * Purpose:  Print a matrix
 * In args:  A, m, n
 */
void print_matrix(float *A, int m, int n)
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
    half *A = nullptr, *B = nullptr;
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
    float *C = new float[m * k];
    cudaEventRecord(start);

    wmma_mat_mult(A, B, C, m, n, k);

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
