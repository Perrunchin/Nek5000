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

// Raptor Headers
#include "core/par_matrix.hpp"
#include "multilevel/par_multilevel.hpp"

// Hypre Headers
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"

// Namespaces
using namespace raptor;

// Global variables
ParMultilevel *amg_preconditioner;

// Functions definitions
void set_amg_preconditioner_()
{
    // Create solver
    double strong_threshold = 0.25;

    amg_preconditioner = new ParMultilevel(A_fem_rap, strong_threshold, CLJP, Classical, SOR);
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

    if (!mass_matrix_precond)
    {
        for (int i = 0; i < num_loc_dofs; i++)
        {
            int row = ranking[i];

            if ((row_start <= row) and (row <= row_end))
            {
                int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

                f_fem_rap[row - row_start] = right_hand_side_vector[idx];
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
                    int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);
                    double Bd_fem_value = Bd_fem_rap[row - row_start];
                    double Binv_sem_value = Binv_sem_rap[row - row_start];

                    f_fem_rap[row - row_start] = Bd_fem_value * Binv_sem_value * right_hand_side_vector[idx];
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
                    int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);
                    double Binv_sem_value = Binv_sem_rap[row - row_start];

                    Bf_fem_rap[row - row_start] = Binv_sem_value * right_hand_side_vector[idx];
                }
            }

            B_fem_rap->mult(Bf_fem_rap, f_fem_rap);
        }
    }

    // Solve preconditioning linear system
    amg_preconditioner->solve(u_fem_rap, f_fem_rap, NULL, 1);
//    amg_preconditioner->set_res_tol(1.0e-7);
//    amg_preconditioner->solve(u_fem_rap, f_fem_rap);

    double u_loc[num_loc_dofs];

    for (int i = 0; i < num_loc_dofs; i++)
    {
        int row = ranking[i];

        if ((row_start <= row) and (row <= row_end))
        {
            u_loc[i] = u_fem_rap[row - row_start];
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
