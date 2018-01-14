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

// Functions definitions
void set_amg_preconditioner_()
{
    // Create AMG preconditioner
    setup_amg_(ia, ja, a, nnz);
}

void mass_matrix_preconditioning_(double *right_hand_side_vector)
{
    /*
     * Updates the rhs when using mass matrix preconditioning
     */
    // Distribute RHS values to their corresponding processors
    int row_start = hypre_ParCSRMatrixFirstRowIndex(A_fem);
    int row_end = hypre_ParCSRMatrixLastRowIndex(A_fem);

    // Prepare RHS after distribution
    bool mass_matrix_precond = false;
    bool mass_diagonal = false;

    if (!mass_matrix_precond)
    {
        return;
    }
    else
    {
        HYPRE_IJVectorInitialize(f_bc);

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

        HYPRE_IJVectorAssemble(f_bc);
    }

    // Distribute values to nodes but don't assemble them
    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            int idx = i + e * (n_x * n_y * n_z);

            right_hand_side_vector[idx] = 0.0;
        }
    }

    int num_rows = row_end - row_start + 1;
    double *visited = mem_alloc<double>(num_rows);

    for (int row = 0; row < num_rows; row++)
    {
        visited[row] = 0.0;
    }

    for (int i = 0; i < num_loc_dofs; i++)
    {
        int row = ranking[i];

        if ((row_start <= row) and (row <= row_end))
        {
            if (visited[row - row_start] == 0.0)
            {
                int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

                HYPRE_IJVectorGetValues(f_bc, 1, &row, &right_hand_side_vector[idx]);

                visited[row - row_start] = 1.0;
            }
        }
    }
}
