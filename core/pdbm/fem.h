/*
 * FEM functions declaration
 */
// Hypre Headers
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// Declarations
#ifndef FEM_H
#define FEM_H

// Global variables declaration
extern HYPRE_IJMatrix A_bc;
extern HYPRE_ParCSRMatrix A_fem;
extern HYPRE_IJMatrix B_bc;
extern HYPRE_ParCSRMatrix B_fem;
extern HYPRE_IJVector f_bc;
extern HYPRE_ParVector f_fem;
extern HYPRE_IJVector u_bc;
extern HYPRE_ParVector u_fem;
extern HYPRE_IJVector Bf_bc;
extern HYPRE_ParVector Bf_fem;
extern double *Binv_sem;
extern double *Bd_fem;
extern int num_nodes;
extern double *gll_nodes;

// Functions declaration
extern "C"
{
    void assemble_fem_matrices_();
    void set_sem_inverse_mass_matrix_(double *);
    void save_fem_matrices_();
}

void mesh_connectivity(double **&, long int **&, int &, int &);
void rectangular_to_triangular(long int **&, int &, long int **, int);
void hexahedral_to_tetrahedral(long int **&, int &, long int **, int);

// FEM Assembly
void generate_basis();
void fem_matrices(double **, long int **, int);

// Math functions
double distance(double *, double *, int);
double interp(double, double *, double *, int);
double determinant(double **, int);
void inverse(double **&, double **, int, double = 0.0);
void matrix_matrix_mul(double **, double **, double **, int, int, int, bool, bool);
void matrix_scaling(double **, double, int, int);
void row_scaling(double **, double **, double *, int, int);

// Geometric functions
double x_map(double, double);
double y_map(double, double);
void J_xp_map(double **, double, double);
double p_map(double, double, double [], double []);
double q_map(double, double, double [], double []);
void J_pr_map(double **, double, double, double [], double []);

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

void print_matrix(double **, int, int);
void print_vertices(double **, int, int);
void print_elements(long int **, int, int);

#endif
