/*
 * Definition of interface functions to connect Fortran with C/C++
 */
// Headers
#include "interface.h"
#include "utilities.h"

// C/C++ Headers
#include <iostream>
#include <limits>
#include <algorithm>
#include "mpi.h"

// Namespaces
using namespace std;

// Global variables definition
int n_x, n_y, n_z, n_elem, n_dim;
double ***mesh;
long **glo_num;
double **press_mask;

// Set functions
void enable_mpi_output_()
{
    /*
     * Creates a file to store each rank's stdout
     */
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string name;
    name = "out_" + to_string(rank) + ".dat";
    freopen(name.c_str(), "w", stdout);
}

void set_element_data_(int &n_x_, int &n_y_, int &n_z_, int &n_elem_, int &n_dim_)
{
    n_x = n_x_;
    n_y = n_y_;
    n_z = n_z_;
    n_elem = n_elem_;
    n_dim = n_dim_;
}

void set_mesh_data_(double *x_m_, double *y_m_, double *z_m_)
{
    double *mesh_[] = { x_m_, y_m_, z_m_ };

    mesh = new double**[n_elem];

    for (int e = 0; e < n_elem; e++)
    {
        mesh[e] = new double*[n_dim];

        for (int d = 0; d < n_dim; d++)
        {
            mesh[e][d] = mesh_[d] + e * (n_x * n_y * n_z);
        }
    }
}

void set_global_numbering_(long *glo_num_)
{
    glo_num = mem_alloc<long>(n_elem, n_x * n_y * n_z);

    for (int e = 0; e < n_elem; e++)
    {
        for (int idx = 0; idx < n_x * n_y * n_z; idx++)
        {
            glo_num[e][idx] = glo_num_[idx + e * (n_x * n_y * n_z)] - 1;
        }
    }
}

void set_pressure_mask_(double* press_mask_)
{
    press_mask = new double*[n_elem];

    for (int e = 0; e < n_elem; e++)
    {
        press_mask[e] = press_mask_ + e * (n_x * n_y * n_z);
    }
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
