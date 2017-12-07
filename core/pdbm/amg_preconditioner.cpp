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
#include <limits>

// Hypre Headers
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"

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
    HYPRE_BoomerAMGSetMaxIter(amg_preconditioner, 1); // maximum number of V-cycles
    HYPRE_BoomerAMGSetTol(amg_preconditioner, 1e-3); // convergence tolerance

    // Setup preconditioner
    HYPRE_BoomerAMGSetup(amg_preconditioner, A_fem, NULL, NULL);
}

void amg_fem_preconditioner_(double *solution_vector, double *right_hand_side_vector)
{
    /*
     * Solves the system $\boldsymbol{M} \boldsymbol{r} = \boldsymbol{z}$ using Algebraic Multigrid
     */
    // Distribute RHS values to their corresponding processors
    int num_rows = hypre_ParCSRMatrixGlobalNumRows(A_fem);
    int row_start = hypre_ParCSRMatrixFirstRowIndex(A_fem);
    int row_end = hypre_ParCSRMatrixLastRowIndex(A_fem);

    // Prepare RHS after distribution
    bool mass_matrix_precond = false;
    bool mass_diagonal = false;

    HYPRE_IJVectorInitialize(f_bc);

    if (!mass_matrix_precond)
    {
        for (int i = 0; i < num_loc_dofs; i++)
        {
            int row = ranking[i];

            if ((row_start <= row) and (row <= row_end))
            {
                int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

                HYPRE_IJVectorSetValues(f_bc, 1, &row, &right_hand_side_vector[idx]);
            }
        }
    }
    else
    {
        if (mass_diagonal)
        {
            for (int i = 0; i < num_loc_dofs; i++)
            {
                int row = ranking[i];

                if ((row_start <= row) and (row <= row_end))
                {
                    double Bd_fem_value;
                    double Binv_sem_value;
                    int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

                    HYPRE_IJVectorGetValues(Bd_bc, 1, &row, &Bd_fem_value);
                    HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &row, &Binv_sem_value);

                    double value = Bd_fem_value * Binv_sem_value * right_hand_side_vector[idx];

                    HYPRE_IJVectorSetValues(f_bc, 1, &row, &value);
                }
            }
        }
        else
        {
            for (int i = 0; i < num_loc_dofs; i++)
            {
                int row = ranking[i];

                if ((row_start <= row) and (row <= row_end))
                {
                    double Binv_sem_value;
                    int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

                    HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &row, &Binv_sem_value);

                    double value = Binv_sem_value * right_hand_side_vector[idx];

                    HYPRE_IJVectorSetValues(Bf_bc, 1, &row, &value);
                }
            }

            HYPRE_ParCSRMatrixMatvec(1.0, B_fem, Bf_fem, 0.0, f_fem);
        }
    }

    HYPRE_IJVectorAssemble(f_bc);

    // Solve preconditioned system
    HYPRE_BoomerAMGSolve(amg_preconditioner, A_fem, f_fem, u_fem);

    double u_loc[num_loc_dofs];

    for (int i = 0; i < num_loc_dofs; i++)
    {
        int row = ranking[i];

        if ((row_start <= row) and (row <= row_end))
        {
            HYPRE_IJVectorGetValues(u_bc, 1, &row, &u_loc[i]);
        }
        else
        {
            u_loc[i] = - numeric_limits<double>::max();
        }
    }

    distribute_data_(u_loc, num_loc_dofs);

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            int idx = i + e * (n_x * n_y * n_z);

            solution_vector[idx] = 0.0;
        }
    }

    for (int i = 0; i < num_loc_dofs; i++)
    {
        int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

        solution_vector[idx] = u_loc[i];
    }
}
