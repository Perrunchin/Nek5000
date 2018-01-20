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
    amg_preconditioner->solve(u_fem_rap, f_fem_rap, NULL, 100);
//    amg_preconditioner->set_res_tol(1.0e-7);
//    amg_preconditioner->solve(u_fem_rap, f_fem_rap);

    double u_loc[num_loc_dofs];
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
                u_loc[i] = u_fem_rap[row - row_start];

                visited[row - row_start] = 1.0;
            }
            else
            {
                u_loc[i] = 0.0;
            }
        }
        else
        {
            u_loc[i] = 0.0;
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

    mem_free<double>(visited, num_rows);
}

// Memory management
template<typename DataType>
DataType* mem_alloc(int n)
{
    DataType *pointer = new DataType[n];

    return pointer;
}

template<typename DataType>
DataType** mem_alloc(int n, int m)
{
    DataType **pointer = new DataType*[n];

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType[m];
    }

    return pointer;
}

template<typename DataType>
DataType*** mem_alloc(int n, int m, int d)
{
    DataType ***pointer = new DataType**[n];

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType*[m];

        for (int j = 0; j < m; j++)
        {
            pointer[i][j] = new DataType[d];
        }
    }

    return pointer;
}

template<typename DataType>
void mem_free(DataType *pointer, int n)
{
    delete[] pointer;
}

template<typename DataType>
void mem_free(DataType **pointer, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        delete[] pointer[i];
    }

    delete[] pointer;
}

template<typename DataType>
void mem_free(DataType ***pointer, int n, int m, int d)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            delete[] pointer[i][j];
        }

        delete[] pointer[i];
    }

    delete[] pointer;
}
