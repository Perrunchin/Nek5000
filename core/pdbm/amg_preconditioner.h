/*
 * Header for solving a linear system with the FEM matrix of the mesh
 */

// Declarations
#ifndef AMG_PRECONDITIONER_H
#define AMG_PRECONDITIONER_H

// Hypre Headers
#include "HYPRE.h"

// Global variables
extern HYPRE_Solver amg_preconditioner;

extern "C"
{
    void set_amg_preconditioner_();
    void setup_amg_(int*, int*, double*, int&);
    void mass_matrix_preconditioning_(double*);
}

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
