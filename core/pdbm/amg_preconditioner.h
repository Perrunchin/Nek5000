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
    void mass_matrix_preconditioning_(double*);
}

#endif
