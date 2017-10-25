/*
 * FEM functions declaration
 */
// Raptor Headers
#include "core/matrix.hpp"
#include "core/vector.hpp"

// Namespaces
using namespace raptor;

// Declarations
#ifndef FEM_H
#define FEM_H

// Global variables declaration
extern CSRMatrix *A_fem;
extern CSRMatrix *B_fem;
extern Vector f_fem;
extern Vector u_fem;
extern Vector Bf_fem;
extern double *Binv_sem;
extern double *Bd_fem;

// Functions declaration
extern "C"
{
    void assemble_fem_matrices_();
    void set_sem_inverse_mass_matrix_(double*);
    void save_fem_matrices_();
}

void mesh_connectivity(double**&, long int**&, int&, int&);
void rectangular_to_triangular(long int**&, int&, long int**, int);
void hexahedral_to_tetrahedral(long int**&, int&, long int**, int);

// FEM Assembly
void fem_matrices(double**, long int**, int);

// Math functions
double determinant(double**, int);
void inverse(double**&, double**, int, double);
void matrix_matrix_mul(double**, double**, double**, int, int, int, bool, bool);
void matrix_scaling(double**, double, int, int);
void row_scaling(double**, double**, double*, int, int);

// Geometric functions
double x_map(double, double);
double y_map(double, double);
double det_J_map(double, double);

// Utility functions
template<typename DataType>
DataType* allocate_single_pointer(int);

template<typename PointerType>
void free_single_pointer(PointerType&);

template<typename DataType>
DataType** allocate_double_pointer(int, int);

template<typename DataType>
DataType** allocate_double_pointer(int, int, double);

template<typename PointerType>
void free_double_pointer(PointerType&, int);

void print_matrix(double**, int, int);
void print_vertices(double**, int, int);
void print_elements(long int**, int, int);
//void save_matrix(std::string, Epetra_CrsMatrix*);

#endif
