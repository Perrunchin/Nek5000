/*
 * Definition of utility functions
 */

// Headers
#include <algorithm>
#include "mpi.h"
#include "utilities.h"

// Namespaces
using namespace std;

// Functions definition
void breakpoint_()
{
    asm("int $3");
}

long maximum_value(long *array, int n)
{
    long max_val = 0;

    for (int i = 0; i < n; i++)
    {
        max_val = max(max_val, array[i]);
    }

    MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    return max_val;
}

long maximum_value(long **array, int n, int m)
{
    long max_val = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            max_val = max(max_val, array[i][j]);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    return max_val;
}

// Memory management
template<typename DataType>
DataType* mem_alloc(int n)
{
    DataType *pointer = new DataType[n]; 

    return pointer;
}

template<typename DataType>
DataType** mem_alloc(int n, int m)
{
    DataType **pointer = new DataType*[n]; 

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType[m];
    }

    return pointer;
}

template<typename DataType>
DataType*** mem_alloc(int n, int m, int d)
{
    DataType ***pointer = new DataType**[n]; 

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType*[m];

        for (int j = 0; j < m; j++)
        {
            pointer[i][j] = new DataType[d];
        }
    }

    return pointer;
}

template<typename DataType>
void mem_free(DataType *pointer, int n)
{
    delete[] pointer;
}

template<typename DataType>
void mem_free(DataType **pointer, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        delete[] pointer[i];
    }

    delete[] pointer;
}

template<typename DataType>
void mem_free(DataType ***pointer, int n, int m, int d)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            delete[] pointer[i][j];
        }

        delete[] pointer[i];
    }

    delete[] pointer;
}
