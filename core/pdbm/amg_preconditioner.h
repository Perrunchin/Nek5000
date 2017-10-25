/*
 * Header for solving a linear system with the FEM matrix of the mesh
 */

// Declarations
#ifndef AMG_PRECONDITIONER_H
#define AMG_PRECONDITIONER_H

// Namespaces
using namespace raptor;

// Raptor Headers
#include "multilevel/multilevel.hpp"

// Global variables
extern Multilevel *amg_preconditioner;

extern "C"
{
    void amg_fem_preconditioner_(double*, double*);
    void set_amg_preconditioner_();
}

#endif
