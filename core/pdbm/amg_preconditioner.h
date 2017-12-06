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
//extern HYPRE_Solver amg_preconditioner;

extern "C"
{
    void amg_fem_preconditioner_(double*, double*);
    void set_amg_preconditioner_();
}

#endif
