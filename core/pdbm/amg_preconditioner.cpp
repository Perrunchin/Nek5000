/*
 * Functions definitionf for AMG preconditioner
 */
// Declaration Header
#include "interface.h"
#include "fem.h"
#include "amg_preconditioner.h"

// C/C++ Headers
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

// Hypre Headers
#include "core/matrix.hpp"
#include "multilevel/multilevel.hpp"

// Namespaces
using namespace raptor;

// Global variables
Multilevel *amg_preconditioner;

// Functions definitions
void set_amg_preconditioner_()
{
    // Create solver
    double strong_threshold = 0.25;

    amg_preconditioner = new Multilevel(A_fem, strong_threshold);
}

void amg_fem_preconditioner_(double* solution_vector, double* right_hand_side_vector)
{
    /*
     * Solves the system $\boldsymbol{M} \boldsymbol{r} = \boldsymbol{z}$ using Algebraic Multigrid
     */
    bool mass_matrix_precond = true;
    bool mass_diagonal = true;

    if (!mass_matrix_precond)
    {
        for (int i = 0; i < max_rank; i++)
        {
            f_fem[i] = right_hand_side_vector[rhs_index[i]];
        }
    }
    else
    {
        if (mass_diagonal)
        {
            for (int i = 0; i < max_rank; i++)
            {
                f_fem[i] = Bd_fem[i] * Binv_sem[rhs_index[i]] * right_hand_side_vector[rhs_index[i]];
            }
        }
        else
        {
            for (int i = 0; i < max_rank; i++)
            {
                Bf_fem[i] = Binv_sem[rhs_index[i]] * right_hand_side_vector[rhs_index[i]];
            }

            B_fem->mult(Bf_fem, f_fem);
        }
    }

    // Solve preconditioned system
    amg_preconditioner->solve(u_fem, f_fem);

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        solution_vector[i] = 0.0;
    }

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (ranking[i] < max_rank)
        {
            solution_vector[i] = u_fem[ranking[i]];
        }
    }
}
