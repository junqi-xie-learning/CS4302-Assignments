/* File:
 *     sum_vect.cpp
 *
 *
 * Idea:
 *     Computes a parallel sum of a vector.  Vectors are distributed by blocks
 *     with the `parallel for` directive, and the variable `sum` is used as a
 *     reduction variable.  The `thread_count` is specified by the user.
 *
 * Compile:
 *     g++ -g -Wall -fopenmp -o sum_vect.out sum_vect.cpp
 * Usage:
 *     ./sum_vect.out <thread_count> <n>
 *
 * Input:
 *     None unless compiled with debug mode.
 *     If in debug mode, read vector `x` from standard input.
 * Output:
 *     Elapsed time for the computation
 *     If in debug mode, print the sum of the vector.
 */

#include <iostream>
#include <random>
#include <omp.h>
using namespace std;

bool debug = false;

/*------------------------------------------------------------------
 * Function:  sum_vect
 * Purpose:   Sum up elements of a vector `x` of size `n`.
 * In args:   x, n, thread_count
 * Out arg:   sum
 */
template <typename T>
T sum_vect(T x[], int n, int thread_count)
{
    T sum{ };
    #pragma omp parallel for num_threads(thread_count) \
        reduction(+: sum)
    for (int i = 0; i < n; i++)
        sum += x[i];
    return sum;
}

/*------------------------------------------------------------------
 * Function: generate_vector
 * Purpose:  Use the random number generator random to generate
 *           the entries in `x` in [0.0, 1.0]
 * In arg:   n
 * Out arg:  x
 */
double *generate_vector(int n)
{
    default_random_engine generator;
    uniform_real_distribution<double> distribution{ 0.0, 1.0 };

    double *x = new double[n];
    for (int i = 0; i < n; i++)
        x[i] = distribution(generator);
    return x;
}

/*------------------------------------------------------------------
 * Function: read_vector
 * Purpose:  Read in a vector
 * In arg:   n
 * Out arg:  x
 */
double *read_vector(int n)
{
    double *x = new double[n];
    for (int i = 0; i < n; i++)
        cin >> x[i];
    return x;
}


int main(int argc, char *argv[])
{
    // Get command line args
    int thread_count = stoi(argv[1]), n = stoi(argv[2]);

    // Generate vector
    double *x = nullptr;
    if (debug)
    {
        cout << "Enter the vector: " << endl;
        x = read_vector(n);
    }
    else
    {
        cout << "Generated vector of size " << n << endl;
        x = generate_vector(n);
    }

    // Call `sum_vect` and get the time elapsed
    double start = omp_get_wtime();
    int sum = sum_vect(x, n, thread_count);
    double finish = omp_get_wtime(), elapsed = finish - start;
    cout << "Sum calculated. Elapsed time: " << elapsed << " seconds" << endl;

    if (debug)
        cout << "The sum is: " << sum << endl;

    delete[] x;
    return 0;
}
