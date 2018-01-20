/*
 * Header for solving a linear system with the FEM matrix of the mesh
 */

// Declarations
#ifndef AMG_PRECONDITIONER_H
#define AMG_PRECONDITIONER_H

// Raptor Headers
#include "multilevel/par_multilevel.hpp"

// Hypre Headers
#include "HYPRE.h"
 
// Namespaces
using namespace raptor;

// Global variables
extern ParMultilevel *amg_preconditioner;

extern "C"
{
    void amg_fem_preconditioner_(double*, double*);
    void set_amg_preconditioner_();
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
