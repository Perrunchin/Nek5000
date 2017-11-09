/*
 * Definition of interface functions to connect Fortran with C/C++
 */
// Headers
#include "interface.h"

// C/C++ Headers
#include <iostream>
#include <limits>
#include <algorithm>

// Global variables definition
int n_x = 0, n_y = 0, n_z = 0, n_elem = 0, n_dim = 0;
double* x_m = NULL;
double* y_m = NULL;
double* z_m = NULL;
long int* glo_num = NULL;
double* press_mask = NULL;
long int* ranking = NULL;
long int max_rank = 0;
double* rhs = NULL;
long int* rhs_index = NULL;
int* indices = NULL;
double lambda = 1.0;
int mapping = 0;

// Set functions
void set_element_data_(int& n_x_, int& n_y_, int& n_z_, int& n_elem_, int& n_dim_)
{
    n_x = n_x_;
    n_y = n_y_;
    n_z = n_z_;
    n_elem = n_elem_;
    n_dim = n_dim_;
}

void set_mesh_data_(double* x_m_, double* y_m_, double* z_m_)
{
    x_m = new double[n_x * n_y * n_z * n_elem];
    y_m = new double[n_x * n_y * n_z * n_elem];
    z_m = new double[n_x * n_y * n_z * n_elem];

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        x_m[i] = x_m_[i];
        y_m[i] = y_m_[i];
        z_m[i] = z_m_[i];
    }
}

void set_global_numbering_(long int* glo_num_)
{
    glo_num = new long int[n_x * n_y * n_z * n_elem];

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        glo_num[i] = glo_num_[i] - 1;
    }
}

void set_pressure_mask_(double* press_mask_)
{
    press_mask = new double[n_x * n_y * n_z * n_elem];

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        press_mask[i] = press_mask_[i];
    }
}

void compute_ranking_()
{
    // Allocate memory
    ranking = new long int[n_x * n_y * n_z * n_elem];

    // Mask the global numbering
    std::vector<std::pair<long int, long int>> vertex_numbering;

    for (long int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (press_mask[i] == 0.0)
        {
            vertex_numbering.push_back(std::pair<long int, long int>(std::numeric_limits<int>::max(), i));
        }
        else
        {
            vertex_numbering.push_back(std::pair<long int, long int>(glo_num[i], i));
        }
    }

    // Compute ranking
    std::sort(vertex_numbering.begin(), vertex_numbering.end());

    long int rank = 1;

    ranking[vertex_numbering[0].second] = 0;

    for (long int i = 1; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (vertex_numbering[i].first == vertex_numbering[i - 1].first)
        {
            ranking[vertex_numbering[i].second] = rank - 1;
        }
        else
        {
            ranking[vertex_numbering[i].second] = rank;
            rank++;
        }
    }

    // Maximum rank
    int max_global = *std::max_element(glo_num, glo_num + (n_x * n_y * n_z * n_elem));
    max_rank = *std::max_element(ranking, ranking + (n_x * n_y * n_z * n_elem));

    if (max_global == max_rank)
    {
        max_rank += 1;
    }

    // Allocate memory for rhs to be filled later
    rhs = new double[max_rank];

    // Compute rhs indexing
    rhs_index = new long int[n_x * n_y * n_z * n_elem];

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (ranking[i] == max_rank)
        {
            rhs_index[ranking[i]] = -1;
            continue;
        }

        rhs_index[ranking[i]] = i;
    }

    // Create indices array
    indices = new int[max_rank];

    for (int i = 0; i < max_rank; i++)
    {
        indices[i] = i;
    }
}

void set_lambda_(double &lambda_)
{
    lambda = lambda_;
}

void set_mapping_(int &mapping_)
{
    mapping = mapping_;
}

// Memory management functions
void free_memory()
{
    free_single_pointer(glo_num);
    free_single_pointer(press_mask);
    free_single_pointer(x_m);
    free_single_pointer(y_m);
    free_single_pointer(z_m);
}

// Displaying functions
void print_variables_()
{
    printf("n_x = %d, n_y = %d, n_z = %d, n_elem = %d, n_dim = %d\n", n_x, n_y, n_z, n_elem, n_dim);

    printf("glo_num = ");

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        printf("%ld ", glo_num[i]);
    }

    printf("\n");

    printf("press_mask = ");

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        printf("%f ", press_mask[i]);
    }

    printf("\n");

    printf("Mesh:\n");

    for (int e = 0; e < n_elem; e++)
    {
        printf("Element: %d\n", e + 1);

        for (int ijk = 0; ijk < n_x * n_y * n_z; ijk++)
        {
            printf("(%f, %f, %f) ", x_m[ijk + e * n_elem], y_m[ijk + e * n_elem], z_m[ijk + e * n_elem]);
        }

        printf("\n");
    }
}

// Utility functions
template<typename PointerType>
void free_single_pointer(PointerType& single_pointer)
{
    delete[] single_pointer;
}

template<typename PointerType>
void free_double_pointer(PointerType& double_pointer, int size)
{
    for (int i = 0; i < size; i++)
    {
        delete[] double_pointer[i];
    }

    delete[] double_pointer;
}
