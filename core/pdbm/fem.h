/*
 * FEM functions declaration
 */
// Headers
#include <vector>
#include <functional>

// Raptor Headers
#include "core/matrix.hpp"
#include "core/vector.hpp"

// Hypre Headers
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// Namespaces
using namespace raptor;
using namespace std;

// Declarations
#ifndef FEM_H
#define FEM_H

// Global variables declaration
//extern CSRMatrix *A_fem;
//extern CSRMatrix *B_fem;
//extern Vector f_fem;
//extern Vector u_fem;
//extern Vector Bf_fem;
//extern double *Binv_sem;
//extern double *Bd_fem;

extern long *ranking;
extern long **dof_map;
extern int num_loc_dofs;
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
extern HYPRE_IJVector Bd_bc;
extern HYPRE_ParVector Bd_fem;
extern HYPRE_IJVector Binv_sem_bc;
extern HYPRE_ParVector Binv_sem;

// Functions declaration
extern "C"
{
    void assemble_fem_matrices_();
    void set_sem_inverse_mass_matrix_(double*);

    // Fortran functions
    void set_amg_gs_handle_(long*, int&);
    void compress_data_(long*, int&);
    void distribute_data_(double*, int&);
}

// FEM Assembly
void fem_matrices();
void quadrature_rule(double**&, double*&, int, int);
void mesh_connectivity(int**&, int**&, int, int);
void x_map(double*&, double*, double**, int, vector<function<double (double*)>>);
void J_xr_map(double**&, double*, double**, int, vector<function<void (double*, double*)>>);
void parallel_ranking(long*&, long*, int, long);
void serial_ranking(long*, long*, int);

// Math functions
double determinant(double**, int);
void inverse(double**&, double**, int);

// Memory management
template<typename DataType>
DataType* mem_alloc(int);

template<typename DataType>
DataType** mem_alloc(int, int);

template<typename DataType>
DataType*** mem_alloc(int, int, int);

template<typename DataType>
void mem_free(DataType*, int);

template<typename DataType>
void mem_free(DataType**, int, int);

template<typename DataType>
void mem_free(DataType***, int, int, int);

#endif
