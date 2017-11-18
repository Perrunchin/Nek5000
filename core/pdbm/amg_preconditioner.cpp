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
    HYPRE_BoomerAMGSetMaxIter(amg_preconditioner, 40); // maximum number of V-cycles
    HYPRE_BoomerAMGSetTol(amg_preconditioner, 1e-2); // convergence tolerance

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

    for (int i = row_start; i <= row_end; i++)
    {
        double zero = 0.0;

        HYPRE_IJVectorSetValues(f_bc, 1, &i, &zero);
    }

    double f_array[num_rows] = { 0 };

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            int idx = i + e * (n_x * n_y * n_z);
            int row = ranking[idx];

            if (row >= 0)
            {
                f_array[row] = right_hand_side_vector[idx];
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, f_array, num_rows, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Prepare RHS after distribution
    bool mass_matrix_precond = false;
    bool mass_diagonal = false;

    if (!mass_matrix_precond)
    {
        for (int i = row_start; i <= row_end; i++)
        {
            double value = f_array[i];

            HYPRE_IJVectorSetValues(f_bc, 1, &i, &value);
        }
    }
    else
    {
        if (mass_diagonal)
        {
            for (int i = row_start; i <= row_end; i++)
            {
                double Bd_fem_value;
                double Binv_sem_value;

                HYPRE_IJVectorGetValues(Bd_bc, 1, &i, &Bd_fem_value);
                HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &i, &Binv_sem_value);

                double value = Bd_fem_value * Binv_sem_value * f_array[i];

                HYPRE_IJVectorSetValues(f_bc, 1, &i, &value);
            }
        }
        else
        {
            for (int i = row_start; i <= row_end; i++)
            {
                double Binv_sem_value;

                HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &i, &Binv_sem_value);

                double value = Binv_sem_value * f_array[i];

                HYPRE_IJVectorSetValues(Bf_bc, 1, &i, &value);
            }

            HYPRE_ParCSRMatrixMatvec(1.0, B_fem, Bf_fem, 0.0, f_fem);
        }
    }

    // Solve preconditioned system
    HYPRE_BoomerAMGSolve(amg_preconditioner, A_fem, f_fem, u_fem);

    double u_array[num_rows] = { 0 };

    for (int i = row_start; i <= row_end; i++)
    {
        HYPRE_IJVectorGetValues(u_bc, 1, &i, &u_array[i]);
    }

    MPI_Allreduce(MPI_IN_PLACE, u_array, num_rows, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (int idx = 0; idx < n_elem * n_x * n_y * n_z; idx++)
    {
        int row = ranking[idx];

        if (row >= 0)
        {
            solution_vector[idx] = u_array[row];
        }
        else
        {
            solution_vector[idx] = 0.0;
        }
    }
}
