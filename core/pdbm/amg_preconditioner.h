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
    void amg_fem_preconditioner_(double*, double*);
    void set_amg_preconditioner_();
}

#endif
