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
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// Global variables
HYPRE_Solver amg_preconditioner;

// Functions definitions
void set_amg_preconditioner_()
{
    // Create solver
    HYPRE_BoomerAMGCreate(&amg_preconditioner);

    // Set some parameters (See Reference Manual for more parameters)
    HYPRE_BoomerAMGSetPrintLevel(amg_preconditioner, 3);  // print solve info + parameters
    HYPRE_BoomerAMGSetOldDefault(amg_preconditioner); // Falgout coarsening with modified classical interpolaiton
    HYPRE_BoomerAMGSetRelaxType(amg_preconditioner, 3); // G-S/Jacobi hybrid relaxation
    HYPRE_BoomerAMGSetRelaxOrder(amg_preconditioner, 1); // uses C/F relaxation
    HYPRE_BoomerAMGSetNumSweeps(amg_preconditioner, 1); // Sweeeps on each level
    HYPRE_BoomerAMGSetMaxLevels(amg_preconditioner, 20); // maximum number of levels
    HYPRE_BoomerAMGSetMaxIter(amg_preconditioner, 40); // maximum number of V-cycles
    HYPRE_BoomerAMGSetTol(amg_preconditioner, 1e-2); // convergence tolerance

    // Setup preconditioner
    HYPRE_BoomerAMGSetup(amg_preconditioner, A_fem, NULL, NULL);
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
            rhs[i] = right_hand_side_vector[rhs_index[i]];
        }

        HYPRE_IJVectorSetValues(f_bc, max_rank, indices, rhs);
    }
    else
    {
        if (mass_diagonal)
        {
            for (int i = 0; i < max_rank; i++)
            {
                rhs[i] = Bd_fem[i] * Binv_sem[rhs_index[i]] * right_hand_side_vector[rhs_index[i]];
            }

            HYPRE_IJVectorSetValues(f_bc, max_rank, indices, rhs);
        }
        else
        {
            for (int i = 0; i < max_rank; i++)
            {
                rhs[i] = Binv_sem[rhs_index[i]] * right_hand_side_vector[rhs_index[i]];
            }

            HYPRE_IJVectorSetValues(Bf_bc, max_rank, indices, rhs);

            HYPRE_ParCSRMatrixMatvec(1.0, B_fem, Bf_fem, 0.0, f_fem);
        }
    }

    // Solve preconditioned system
    HYPRE_BoomerAMGSolve(amg_preconditioner, A_fem, f_fem, u_fem);

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        solution_vector[i] = 0.0;
    }

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (ranking[i] < max_rank)
        {
            HYPRE_IJVectorGetValues(u_bc, 1, (int*)(ranking + i), solution_vector + i);
        }
    }
}
