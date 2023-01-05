/* File:
 *     mat_mult.cu
 *
 *
 * Idea:
 *     Computes a product of matrices with Tensor Core simulator.  The simulator
 *     only simulates the computation of a single warp in the GPU.  The matrix
 *     will be split into tiles of 16x16 elements, and each tile will be computed
 *     by a single warp.
 * 
 * Compile:
 *     g++ -g -Wall -o mat_mult.out mat_mult.cpp tensor_core.cpp
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
#include <chrono>
#include "tensor_core.h"
using namespace std;

bool debug = false;

/*------------------------------------------------------------------
 * Function: generate_matrix
 * Purpose:  Use the random number generator random to generate
 *           the entries in `A` in [0.0, 1.0]
 * In arg:   m, n
 * Out arg:  A
 */
__half *generate_matrix(int m, int n)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution{ 0.0, 1.0 };

    __half *A = new __half[m * n];
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
__half *read_matrix(int m, int n)
{
    __half *A = new __half[m * n];
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
    __half *A = nullptr, *B = nullptr;
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

    // Call `mat_mult` and get the time elapsed
    float *C = new float[m * k];
    auto start = chrono::high_resolution_clock::now();

    gemm(A, B, C, m, n, k);

    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Product calculated. Elapsed time: " << elapsed.count() << " seconds" << endl;

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
