/* File:
 *     consumer_producer.cpp
 *
 *
 * Idea:
 *     Simulate a consumer-producer problem.  Several producers are producing
 *     products to a FIFO pipe while some consumers are consuming them on the
 *     other side.  The pipe is implemented as a queue, and the `critical`
 *     directive is used to protect the queue.  To better simulate the reality,
 *     each consumer will sleep for a random time according to the contents.
 *     The simulation will run forever.
 *
 * Compile:
 *     g++ -g -Wall -fopenmp -o consumer_producer consumer_producer.cpp
 * Usage:
 *     ./consumer_producer.out <capacity> <producer> <consumer>
 *
 * Input:
 *     None
 * Output:
 *     Source, destination and contents of each message produced and received.
 */
#include <iostream>
#include <random>
#include <queue>
#include <omp.h>
#include <unistd.h>
using namespace std;

/*------------------------------------------------------------------
 * Function:  producer
 * Purpose:   Routine for the producer threads.  Each producer will keep
 *            producing message of random contents to the queue.
 * In args:   queue, capacity
 * Out arg:   None
 */
void producer(queue<int> &queue, unsigned int capacity)
{
    int current_thread = omp_get_thread_num();
    default_random_engine generator{static_cast<unsigned int>(current_thread)};
    uniform_int_distribution<int> distribution{ 1, 10 };

    while (true)
    {
        while (queue.size() >= capacity); // Wait for a slot in the queue

        #pragma omp critical
        if (queue.size() < capacity)
        {
            // Confirm that no one has filled the slot during the execution.
            int message = distribution(generator);
            queue.push(message);
            cout << "Producer " << current_thread << " produced " << message << endl;
        }
    }
}

/*------------------------------------------------------------------
 * Function:  consumer
 * Purpose:   Routine for the consumer threads.  Each consumer will keep
 *            consuming message from the queue, and sleep corresponding time.
 * In args:   queue, capacity
 * Out arg:   None
 */
void consumer(queue<int> &queue, unsigned int capacity)
{
    int current_thread = omp_get_thread_num();

    while (true)
    {
        while (queue.size() == 0); // Wait for a message in the queue

        int message = 0;
        #pragma omp critical
        if (queue.size() != 0)
        {
            // Confirm that no one has consumed the message during the execution.
            message = queue.front();
            queue.pop();
            cout << "Consumer " << current_thread << " consumed " << message << endl;
        }

        sleep(message);
    }
}


int main(int argc, char *argv[])
{
    // Get command line args
    int capacity = stoi(argv[1]), num_producer = stoi(argv[2]),
        num_consumer = stoi(argv[3]);

    queue<int> queue;

    // Dispatcher for creating producers and consumers
    #pragma omp parallel num_threads(num_producer + num_consumer) shared(queue)
    {
        int current_thread = omp_get_thread_num();
        if (current_thread < num_producer)
            producer(queue, capacity);
        else
            consumer(queue, capacity);
    }
}