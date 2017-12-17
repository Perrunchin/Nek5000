/*
 * FEM functions definition
 */
// Declaration Headers
#include "fem.h"
#include "interface.h"
#include "utilities.h"

// C/C++ Headers
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <algorithm>
#include <iomanip>

// Raptor Headers
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

// Hypre Headers
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"

// Namespaces
using namespace std;
using namespace raptor;

// Global variables definition
ParCSRMatrix *A_fem_rap;
ParCSRMatrix *B_fem_rap;
ParVector f_fem_rap;
ParVector u_fem_rap;
ParVector Bf_fem_rap;
ParVector Binv_sem_rap;
ParVector Bd_fem_rap;

long *ranking;
long **dof_map;
int num_loc_dofs;
HYPRE_IJMatrix A_bc;
HYPRE_ParCSRMatrix A_fem;
HYPRE_IJMatrix B_bc;
HYPRE_ParCSRMatrix B_fem;
HYPRE_IJVector Bd_bc;
HYPRE_ParVector Bd_fem;
HYPRE_IJVector Binv_sem_bc;
HYPRE_ParVector Binv_sem;

// Structures
struct VertexID
{
    int key;
    long value;
    long ranking;
};

// Functions definition
void assemble_fem_matrices_()
{
    // Generate FEM Matrix
    fem_matrices();

    int num_glo_rows = hypre_ParCSRMatrixGlobalNumRows(A_fem);
    int num_glo_cols = num_glo_rows;
    int row_start = hypre_ParCSRMatrixFirstRowIndex(A_fem);
    int row_end = hypre_ParCSRMatrixLastRowIndex(A_fem);
    int num_loc_rows = row_end - row_start + 1;
    int num_loc_cols = num_loc_rows;

    // Transform to RAPtor
    ParCOOMatrix *A_temp = new ParCOOMatrix(num_glo_rows, num_glo_cols, num_loc_rows, num_loc_cols, row_start, row_start);
    ParCOOMatrix *B_temp = new ParCOOMatrix(num_glo_rows, num_glo_cols, num_loc_rows, num_loc_cols, row_start, row_start);

    for (int row = row_start; row <= row_end; row++)
    {
        // Add values
        int num_cols;
        int* cols;
        double* values;

        HYPRE_ParCSRMatrixGetRow(A_fem, row, &num_cols, &cols, &values);

        for (int j = 0; j < num_cols; j++)
        {
            A_temp->add_global_value(row, cols[j], values[j]);
        }

        HYPRE_ParCSRMatrixRestoreRow(A_fem, row, &num_cols, &cols, &values);

        HYPRE_ParCSRMatrixGetRow(B_fem, row, &num_cols, &cols, &values);

        for (int j = 0; j < num_cols; j++)
        {
            B_temp->add_global_value(row, cols[j], values[j]);
        }

        HYPRE_ParCSRMatrixRestoreRow(B_fem, row, &num_cols, &cols, &values);
    }

    A_temp->finalize();
    B_temp->finalize();

    A_fem_rap = new ParCSRMatrix(num_glo_rows, num_glo_cols, num_loc_rows, num_loc_cols, row_start, row_start);
    B_fem_rap = new ParCSRMatrix(num_glo_rows, num_glo_cols, num_loc_rows, num_loc_cols, row_start, row_start);

    A_fem_rap->copy(A_temp);
    B_fem_rap->copy(B_temp);

    delete A_temp;
    delete B_temp;

    u_fem_rap = ParVector(num_glo_rows, num_loc_rows, row_start);
    f_fem_rap = ParVector(num_glo_rows, num_loc_rows, row_start);
    Bf_fem_rap = ParVector(num_glo_rows, num_loc_rows, row_start);
    Bd_fem_rap = ParVector(num_glo_rows, num_loc_rows, row_start);

    for (int row = row_start; row <= row_end; row++)
    {
        double Bd_fem_value;
        double Binv_sem_value;

        HYPRE_IJVectorGetValues(Bd_bc, 1, &row, &Bd_fem_value);

        Bd_fem_rap[row - row_start] = Bd_fem_value;
    }
}

// FEM Assembly
void fem_matrices()
{
    /*
     * Assembles the FEM matrices from SEM meshes
     *
     * Returns A_fem and B_fem
     */
    // MPI Information
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find the maximum vertex id which indicates the number of rows in the full FEM matrices
    long num_vert = maximum_value(glo_num, n_elem, n_x * n_y * n_z) + 1;

    // Assemble full FEM matrices without boundary conditions
    long num_rows = (num_vert + (size - 1)) / size;
    long idx_start = rank * num_rows;
    long idx_end = (rank + 1) * num_rows - 1;

    if (rank == size - 1)
    {
        idx_end = num_vert - 1;
    }

    num_rows = (idx_end + 1) - idx_start;

    HYPRE_IJMatrix A_f;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, idx_start, idx_end, idx_start, idx_end, &A_f);
    HYPRE_IJMatrixSetObjectType(A_f, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_f);

    HYPRE_IJMatrix B_f;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, idx_start, idx_end, idx_start, idx_end, &B_f);
    HYPRE_IJMatrixSetObjectType(B_f, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(B_f);

    // Set quadrature rule
    int n_quad = (n_dim == 2) ? 3 : 4;
    double **q_r;
    double *q_w;

    quadrature_rule(q_r, q_w, n_quad, n_dim);

    // Mesh connectivity
    int num_fem = (n_dim == 2) ? 4 : 8;
    int **v_coord;
    int **t_map;

    mesh_connectivity(v_coord, t_map, num_fem, n_dim);

    // Finite element assembly
    double **A_loc = mem_alloc<double>(n_dim + 1, n_dim + 1);
    double **B_loc = mem_alloc<double>(n_dim + 1, n_dim + 1);
    double **J_xr = mem_alloc<double>(n_dim, n_dim);
    double **J_rx = mem_alloc<double>(n_dim, n_dim);
    double **x_t = mem_alloc<double>(n_dim, n_dim + 1);
    double *q_x = mem_alloc<double>(n_dim);

    vector<function<double (double*)>> phi;
    vector<function<void (double*, double*)>> dphi;

    if (n_dim == 2)
    {
        phi.push_back([] (double *r) { return r[0]; });
        phi.push_back([] (double *r) { return r[1]; });
        phi.push_back([] (double *r) { return 1.0 - r[0] - r[1]; });

        dphi.push_back([] (double *dp, double *r) { dp[0] = 1.0; dp[1] = 0.0; });
        dphi.push_back([] (double *dp, double *r) { dp[0] = 0.0; dp[1] = 1.0; });
        dphi.push_back([] (double *dp, double *r) { dp[0] = -1.0; dp[1] = -1.0; });
    }
    else
    {
        phi.push_back([] (double *r) { return r[0]; });
        phi.push_back([] (double *r) { return r[1]; });
        phi.push_back([] (double *r) { return r[2]; });
        phi.push_back([] (double *r) { return 1.0 - r[0] - r[1] - r[2]; });

        dphi.push_back([] (double *dp, double *r) { dp[0] = 1.0; dp[1] = 0.0; dp[2] = 0.0; });
        dphi.push_back([] (double *dp, double *r) { dp[0] = 0.0; dp[1] = 1.0; dp[2] = 0.0; });
        dphi.push_back([] (double *dp, double *r) { dp[0] = 0.0; dp[1] = 0.0; dp[2] = 1.0; });
        dphi.push_back([] (double *dp, double *r) { dp[0] = -1.0; dp[1] = -1.0; dp[2] = -1.0; });
    }

    int E_x = n_x - 1;
    int E_y = n_y - 1;
    int E_z = (n_dim == 2) ? 1 : n_z - 1;

    for (int e = 0; e < n_elem; e++)
    {
        // Cycle through collocated quads/hexes
        for (int s_z = 0; s_z < E_z; s_z++)
        {
            for (int s_y = 0; s_y < E_y; s_y++)
            {
                for (int s_x = 0; s_x < E_x; s_x++)
                {
                    // Get indices
                    int s[n_dim];

                    if (n_dim == 2)
                    {
                        s[0] = s_x;
                        s[1] = s_y;
                    }
                    else
                    {
                        s[0] = s_x;
                        s[1] = s_y;
                        s[2] = s_z;
                    }

                    int idx[int(pow(2, n_dim))] = { 0 };

                    for (int i = 0; i < pow(2, n_dim); i++)
                    {
                        for (int d = 0; d < n_dim; d++)
                        {
                            idx[i] += (s[d] + v_coord[i][d]) * pow(n_x, d);
                        }
                    }

                    // Cycle through collocated triangles/tets
                    for (int t = 0; t < num_fem; t++)
                    {
                        // Get vertices
                        for (int i = 0; i < n_dim + 1; i++)
                        {
                            for (int d = 0; d < n_dim; d++)
                            {
                                x_t[d][i] = mesh[e][d][idx[t_map[t][i]]];
                            }
                        }

                        // Local FEM matrices
                        // Reset local stiffness and mass matrices
                        for (int i = 0; i < n_dim + 1; i++)
                        {
                            for (int j = 0; j < n_dim + 1; j++)
                            {
                                A_loc[i][j] = 0.0;
                                B_loc[i][j] = 0.0;
                            }
                        }

                        // Build local stiffness matrices by applying quadrature rules
                        for (int q = 0; q < n_quad; q++)
                        {
                            // From r to x
                            x_map(q_x, q_r[q], x_t, n_dim, phi);
                            J_xr_map(J_xr, q_r[q], x_t, n_dim, dphi);
                            inverse(J_rx, J_xr, n_dim);
                            double det_J_xr = determinant(J_xr, n_dim);

                            // Integrand
                            for (int i = 0; i < n_dim + 1; i++)
                            {
                                for (int j = 0; j < n_dim + 1; j++)
                                {
                                    double func = 0.0;

                                    for (int alpha = 0; alpha < n_dim; alpha++)
                                    {
                                        double a = 0.0, b = 0.0;

                                        for (int beta = 0; beta < n_dim; beta++)
                                        {
                                            double dp[n_dim];

                                            dphi[i](dp, q_r[q]);
                                            a += dp[beta] * J_rx[beta][alpha];

                                            dphi[j](dp, q_r[q]);
                                            b += dp[beta] * J_rx[beta][alpha];
                                        }

                                        func += a * b;
                                    }

                                    A_loc[i][j] += func * det_J_xr * q_w[q];
                                    B_loc[i][j] += phi[i](q_r[q]) * phi[j](q_r[q]) * det_J_xr * q_w[q];
                                }
                            }
                        }

                        // Add to global matrix
                        for (int i = 0; i < n_dim + 1; i++)
                        {
                            for (int j = 0; j < n_dim + 1; j++)
                            {
                                int row = glo_num[e][idx[t_map[t][i]]];
                                int col = glo_num[e][idx[t_map[t][j]]];

                                double A_val = A_loc[i][j];
                                double B_val = B_loc[i][j];

                                int ncols = 1;
                                int insert_error;

                                if (std::abs(A_val) > 1.0e-14)
                                {
                                    insert_error = HYPRE_IJMatrixAddToValues(A_f, 1, &ncols, &row, &col, &A_val);
                                }

                                if (std::abs(B_val) > 1.0e-14)
                                {
                                    insert_error = HYPRE_IJMatrixAddToValues(B_f, 1, &ncols, &row, &col, &B_val);
                                }

                                if (insert_error != 0)
                                {
                                    printf("There was an error with entry A(%d, %d) = %f or B(%d, %d) = %f\n", row, col, A_val, row, col, B_val);
                                    exit(EXIT_FAILURE);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    HYPRE_IJMatrixAssemble(A_f);
    HYPRE_IJMatrixAssemble(B_f);

    // Rank vertices after boundary conditions are removed
    num_loc_dofs = 0;

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            if (press_mask[e][i] > 0.0)
            {
                num_loc_dofs++;
            }
        }
    }

    dof_map = mem_alloc<long>(3, num_loc_dofs);
    int idx = 0;

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            if (press_mask[e][i] > 0.0)
            {
                dof_map[0][idx] = i;
                dof_map[1][idx] = e;
                dof_map[2][idx] = glo_num[e][i];
                idx++;
            }
        }
    }

    for (int i = 0; i < num_loc_dofs; i++)
    {
        dof_map[2][i] += 1;
    }

    set_amg_gs_handle_(dof_map[2], num_loc_dofs);

    for (int i = 0; i < num_loc_dofs; i++)
    {
        dof_map[2][i] -= 1;
    }

    long *compression = mem_alloc<long>(num_loc_dofs);
    long offset = 0;

    if (rank < size - 1)
    {
        MPI_Send(&num_loc_dofs, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0)
    {
        MPI_Recv(&offset, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD, NULL);
    }

    MPI_Scan(MPI_IN_PLACE, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < num_loc_dofs; i++)
    {
        compression[i] = offset + i;
    }

    compress_data_(compression, num_loc_dofs);

    ranking = mem_alloc<long>(num_loc_dofs);

//    long max_global = max(num_vert, (long)(num_loc_dofs));
//
//    parallel_ranking(ranking, compression, num_loc_dofs, max_global);

    // Gather data in one proc
    // TODO: Use parallel ranking
    int total_receive[size];
    int displs[size] = { 0 };

    MPI_Gather(&num_loc_dofs, 1, MPI_INT, total_receive, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_count = total_receive[0];

    for (int r = 1; r < size; r++)
    {
        displs[r] = displs[r - 1] + total_receive[r - 1];
        total_count += total_receive[r];
    }

    long *array = NULL;
    long *new_ranking = NULL;

    if (rank == 0)
    {
        array = mem_alloc<long>(total_count);
    }

    MPI_Gatherv(compression, num_loc_dofs, MPI_LONG, array, total_receive, displs, MPI_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        new_ranking = mem_alloc<long>(total_count);
        serial_ranking(new_ranking, array, total_count);
    }

    MPI_Scatterv(new_ranking, total_receive, displs, MPI_LONG, ranking, num_loc_dofs, MPI_LONG, 0, MPI_COMM_WORLD);

    mem_free<long>(array, total_count);
    // END TODO

    // Number of unique vertices after boundary conditions are applied
    long num_vert_bc = maximum_value(ranking, num_loc_dofs) + 1;
    long scan_offset;

    long idx_start_bc = 0;
    long idx_end_bc = 0;

    for (int i = 0; i < num_loc_dofs; i++)
    {
        idx_end_bc = max(idx_end_bc, ranking[i]);
    }

    if (rank < size - 1)
    {
        MPI_Send(&idx_end_bc, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0)
    {
        MPI_Recv(&idx_start_bc, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, NULL);

        idx_start_bc += 1;
    }

    // Assemble FE matrices with boundaries removed
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, idx_start_bc, idx_end_bc, idx_start_bc, idx_end_bc, &A_bc);
    HYPRE_IJMatrixSetObjectType(A_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_bc);

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, idx_start_bc, idx_end_bc, idx_start_bc, idx_end_bc, &B_bc);
    HYPRE_IJMatrixSetObjectType(B_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(B_bc);

    HYPRE_ParCSRMatrix A_f_csr;
    HYPRE_ParCSRMatrix B_f_csr;
    HYPRE_IJMatrixGetObject(A_f, (void**) &A_f_csr);
    HYPRE_IJMatrixGetObject(B_f, (void**) &B_f_csr);

    // TODO: Notify everyone of what rows are to be removed
    long *glo_map = mem_alloc<long>(num_vert);

    for (int i = 0; i < num_vert; i++)
    {
        glo_map[i] = -1;
    }

    for (int i = 0; i < num_loc_dofs; i++)
    {
        glo_map[dof_map[2][i]] = ranking[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, glo_map, num_vert, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    double *glo_press_mask = mem_alloc<double>(num_vert);

    for (int i = 0; i < num_vert; i++)
    {
        glo_press_mask[i] = -1.0;
    }

    for (int e = 0; e < n_elem; e++)
    {
        for (int i = 0; i < n_x * n_y * n_z; i++)
        {
            glo_press_mask[glo_num[e][i]] = press_mask[e][i];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, glo_press_mask, num_vert, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // END TODO:

    // Remove rows and move remaining rows to respective position
    for (int i = idx_start; i <= idx_end; i++)
    {
        if (glo_press_mask[i] > 0.0)
        {
            // Find where this row goes
            int row = glo_map[i];

            // Add values
            int n_cols;
            int* cols;
            double* values;

            HYPRE_ParCSRMatrixGetRow(A_f_csr, i, &n_cols, &cols, &values);

            for (int j = 0; j < n_cols; j++)
            {
                if (glo_press_mask[cols[j]] > 0.0)
                {
                    int col = glo_map[cols[j]];
                    int num_cols = 1;
                    int insert_error = HYPRE_IJMatrixAddToValues(A_bc, 1, &num_cols, &row, &col, values + j);

                    if (insert_error != 0)
                    {
                        printf("There was an error with entry A_fem(%d, %d) = %f\n", row, col, values[j]);
                        exit(EXIT_FAILURE);
                    }
                }
            }

            HYPRE_ParCSRMatrixRestoreRow(A_f_csr, i, &n_cols, &cols, &values);

            HYPRE_ParCSRMatrixGetRow(B_f_csr, i, &n_cols, &cols, &values);

            for (int j = 0; j < n_cols; j++)
            {
                if (glo_press_mask[cols[j]] > 0.0)
                {
                    int col = glo_map[cols[j]];
                    int num_cols = 1;
                    int insert_error = HYPRE_IJMatrixAddToValues(B_bc, 1, &num_cols, &row, &col, values + j);

                    if (insert_error != 0)
                    {
                        printf("There was an error with entry B_fem(%d, %d) = %f\n", row, col, values[j]);
                        exit(EXIT_FAILURE);
                    }
                }
            }

            HYPRE_ParCSRMatrixRestoreRow(B_f_csr, i, &n_cols, &cols, &values);
        }
    }

    HYPRE_IJMatrixAssemble(A_bc);
    HYPRE_IJMatrixGetObject(A_bc, (void**) &A_fem);

    HYPRE_IJMatrixAssemble(B_bc);
    HYPRE_IJMatrixGetObject(B_bc, (void**) &B_fem);

    // Build diagonal mass matrix with full mass matrix without boundary conditions
    double *Bd_sum = mem_alloc<double>(num_rows);

    for (int i = idx_start; i <= idx_end; i++)
    {
        int n_cols;
        int* cols;
        double* values;

        HYPRE_ParCSRMatrixGetRow(B_f_csr, i, &n_cols, &cols, &values);

        Bd_sum[i - idx_start] = 0.0;

        for (int j = 0; j < n_cols; j++)
        {
            Bd_sum[i - idx_start] += values[j];
        }

        HYPRE_ParCSRMatrixRestoreRow(B_f_csr, i, &n_cols, &cols, &values);
    }

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, idx_start_bc, idx_end_bc, &Bd_bc);
    HYPRE_IJVectorSetObjectType(Bd_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Bd_bc);

    for (int i = idx_start; i <= idx_end; i++)
    {
        int row = glo_map[i];

        if (glo_press_mask[i] > 0.0)
        {
            HYPRE_IJVectorAddToValues(Bd_bc, 1, &row, &Bd_sum[i - idx_start]);
        }
    }

    HYPRE_IJVectorAssemble(Bd_bc);
    HYPRE_IJVectorGetObject(Bd_bc, (void**) &Bd_fem);

//    // OUTPUT
//    HYPRE_IJMatrixPrint(A_f, "A_f");
//    HYPRE_IJMatrixPrint(A_bc, "A_bc");
//    HYPRE_IJMatrixPrint(B_f, "B_f");
//    HYPRE_IJMatrixPrint(B_bc, "B_bc");
//
//    if (rank == 0)
//    {
//        ofstream file;
//        file.open("mapping.dat");
//
//        for (int i = 0; i < num_vert; i++)
//        {
//            if (glo_press_mask[i] > 0.0)
//            {
//                file << i << " " << glo_map[i] << endl;
//            }
//        }
//
//        file.close();
//    }
//
//    HYPRE_IJVectorPrint(Bd_bc, "Bd_bc");
//    // END OUTPUT

    // Free memory
    mem_free<double>(q_r, n_quad, n_dim);
    mem_free<double>(q_w, n_quad);
    mem_free<int>(v_coord, pow(n_dim, 2), n_dim);
    mem_free<int>(t_map, num_fem, n_dim + 1);
    mem_free<double>(A_loc, n_dim + 1, n_dim + 1);
    mem_free<double>(B_loc, n_dim + 1, n_dim + 1);
    mem_free<double>(J_xr, n_dim, n_dim);
    mem_free<double>(J_rx, n_dim, n_dim);
    mem_free<double>(x_t, n_dim, n_dim);
    mem_free<double>(q_x, n_dim);
    mem_free<long>(compression, num_loc_dofs);
    mem_free<long>(glo_map, num_vert);
    mem_free<double>(glo_press_mask, num_vert);
    mem_free<double>(Bd_sum, num_rows);
    HYPRE_IJMatrixDestroy(A_f);
    HYPRE_IJMatrixDestroy(B_f);
}

void set_sem_inverse_mass_matrix_(double* inv_B)
{
    /*
     * Build parallel vector of inverse of SEM mass matrix
     */
    int num_glo_rows = hypre_ParCSRMatrixGlobalNumRows(A_fem);
    int row_start = hypre_ParCSRMatrixFirstRowIndex(A_fem);
    int row_end = hypre_ParCSRMatrixLastRowIndex(A_fem);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, row_start, row_end, &Binv_sem_bc);
    HYPRE_IJVectorSetObjectType(Binv_sem_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Binv_sem_bc);
    HYPRE_IJVectorGetObject(Binv_sem_bc, (void**) &Binv_sem);

    HYPRE_IJVector total_count;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, row_start, row_end, &total_count);
    HYPRE_IJVectorSetObjectType(total_count, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(total_count);

    for (int i = 0; i < num_loc_dofs; i++)
    {
        double one = 1.0;
        int row = (int)(ranking[i]);

        if (row >= 0)
        {
            int idx = dof_map[0][i] + dof_map[1][i] * (n_x * n_y * n_z);

            HYPRE_IJVectorAddToValues(Binv_sem_bc, 1, &row, &inv_B[idx]);
            HYPRE_IJVectorAddToValues(total_count, 1, &row, &one);
        }
    }

    HYPRE_IJVectorAssemble(Binv_sem_bc);
    HYPRE_IJVectorAssemble(total_count);

    HYPRE_IJVectorInitialize(Binv_sem_bc);

    for (int i = row_start; i <= row_end; i++)
    {
        double value;
        double count;

        HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &i, &value);
        HYPRE_IJVectorGetValues(total_count, 1, &i, &count);

        if (count > 0.0)
        {
            double new_value = value / count;

            HYPRE_IJVectorSetValues(Binv_sem_bc, 1, &i, &new_value);
        }
    }

    HYPRE_IJVectorAssemble(Binv_sem_bc);
    HYPRE_IJVectorDestroy(total_count);

    // Raptor vectors
    int num_loc_rows = row_end - row_start + 1;
    Binv_sem_rap = ParVector(num_glo_rows, num_loc_rows, row_start);

    for (int row = row_start; row <= row_end; row++)
    {
        double Binv_sem_value;

        HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &row, &Binv_sem_value);

        Binv_sem_rap[row - row_start] = Binv_sem_value;
    }

//    // OUTPUT
//    long num_vert = maximum_value(glo_num, n_elem, n_x * n_y * n_z) + 1;
//    double Binv[num_vert];
//
//    for (int i = 0; i < num_vert; i++)
//    {
//        Binv[i] = 0.0;
//    }
//
//    for (int e = 0; e < n_elem; e++)
//    {
//        for (int i = 0; i < n_x * n_y * n_z; i++)
//        {
//            int idx = i + e * (n_x * n_y * n_z);
//
//            Binv[glo_num[e][i]] = inv_B[idx];
//        }
//    }
//
//    MPI_Allreduce(MPI_IN_PLACE, &Binv, num_vert, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    if (rank == 0)
//    {
//        ofstream file;
//        file.open("Binv_sem.dat");
//
//        for (int i = 0; i < num_vert; i++)
//        {
//            file << fixed << setprecision(16) << Binv[i] << endl;
//        }
//
//        file.close();
//    }
//
//    HYPRE_IJVectorPrint(Binv_sem_bc, "Binv_sem_bc");
//    // END OUTPUT
}

void quadrature_rule(double **&q_r, double *&q_w, int n_quad, int n_dim)
{
    q_r = mem_alloc<double>(n_quad, n_dim);
    q_w = mem_alloc<double>(n_quad);

    if (n_dim == 2)
    {
        if (n_quad == 3)
        {
            q_r[0][0] = 1.0 / 6.0; q_r[0][1] = 1.0 / 6.0;
            q_r[1][0] = 2.0 / 3.0; q_r[1][1] = 1.0 / 6.0;
            q_r[2][0] = 1.0 / 6.0; q_r[2][1] = 2.0 / 3.0;

            q_w[0] = 1.0 / 6.0;
            q_w[1] = 1.0 / 6.0;
            q_w[2] = 1.0 / 6.0;
        }
        else if (n_quad == 4)
        {
            q_r[0][0] = 1.0 / 3.0; q_r[0][1] = 1.0 / 3.0;
            q_r[1][0] = 1.0 / 5.0; q_r[1][1] = 3.0 / 5.0;
            q_r[2][0] = 1.0 / 5.0; q_r[2][1] = 1.0 / 5.0;
            q_r[3][0] = 3.0 / 5.0; q_r[3][1] = 1.0 / 5.0;

            q_w[0] = - 27.0 / 96.0;
            q_w[1] = 25.0 / 96.0;
            q_w[2] = 25.0 / 96.0;
            q_w[3] = 25.0 / 96.0;
        }
        else
        {
            printf("No quadrature rule for %d points available\n", n_quad);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (n_quad == 4)
        {
            double a = (5.0 + 3.0 * sqrt(5.0)) / 20.0;
            double b = (5.0 - sqrt(5.0)) / 20.0;

            q_r[0][0] = a; q_r[0][1] = b; q_r[0][2] = b;
            q_r[1][0] = b; q_r[1][1] = a; q_r[1][2] = b;
            q_r[2][0] = b; q_r[2][1] = b; q_r[2][2] = a;
            q_r[3][0] = b; q_r[3][1] = b; q_r[3][2] = b;

            q_w[0] = 1.0 / 24.0;
            q_w[1] = 1.0 / 24.0;
            q_w[2] = 1.0 / 24.0;
            q_w[3] = 1.0 / 24.0;
        }
        else if (n_quad == 5)
        {
            q_r[0][0] = 1.0 / 2.0; q_r[0][1] = 1.0 / 6.0; q_r[0][2] = 1.0 / 6.0;
            q_r[1][0] = 1.0 / 6.0; q_r[1][1] = 1.0 / 2.0; q_r[1][2] = 1.0 / 6.0;
            q_r[2][0] = 1.0 / 6.0; q_r[2][1] = 1.0 / 6.0; q_r[2][2] = 1.0 / 2.0;
            q_r[3][0] = 1.0 / 6.0; q_r[3][1] = 1.0 / 6.0; q_r[3][2] = 1.0 / 6.0;
            q_r[4][0] = 1.0 / 4.0; q_r[4][1] = 1.0 / 4.0; q_r[4][2] = 1.0 / 4.0;

            q_w[0] = 9.0 / 20.0;
            q_w[1] = 9.0 / 20.0;
            q_w[2] = 9.0 / 20.0;
            q_w[3] = 9.0 / 20.0;
            q_w[4] = - 4.0 / 5.0;
        }
        else
        {
            printf("No quadrature rule for %d points available\n", n_quad);
            exit(EXIT_FAILURE);
        }
    }
}

void mesh_connectivity(int **&v_coord, int **&t_map, int num_fem, int n_dim)
{
    v_coord = mem_alloc<int>(pow(n_dim, 2), n_dim);
    t_map = mem_alloc<int>(num_fem, n_dim + 1);

    if (n_dim == 2)
    {
        v_coord[0][0] = 0; v_coord[0][1] = 0;
        v_coord[1][0] = 1; v_coord[1][1] = 0;
        v_coord[2][0] = 0; v_coord[2][1] = 1;
        v_coord[3][0] = 1; v_coord[3][1] = 1;

        if (num_fem == 2)
        {
            t_map[0][0] = 0; t_map[0][1] = 1; t_map[0][2] = 3;
            t_map[1][0] = 0; t_map[1][1] = 3; t_map[1][2] = 2;
        }
        else if (num_fem == 4)
        {
            t_map[0][0] = 1; t_map[0][1] = 2; t_map[0][2] = 0;
            t_map[1][0] = 3; t_map[1][1] = 0; t_map[1][2] = 1;
            t_map[2][0] = 0; t_map[2][1] = 3; t_map[2][2] = 2;
            t_map[3][0] = 2; t_map[3][1] = 1; t_map[3][2] = 3;
        }
        else
        {
            printf("Wrong number of triangles\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        v_coord[0][0] = 0; v_coord[0][1] = 0; v_coord[0][2] = 0;
        v_coord[1][0] = 1; v_coord[1][1] = 0; v_coord[1][2] = 0;
        v_coord[2][0] = 0; v_coord[2][1] = 1; v_coord[2][2] = 0;
        v_coord[3][0] = 1; v_coord[3][1] = 1; v_coord[3][2] = 0;
        v_coord[4][0] = 0; v_coord[4][1] = 0; v_coord[4][2] = 1;
        v_coord[5][0] = 1; v_coord[5][1] = 0; v_coord[5][2] = 1;
        v_coord[6][0] = 0; v_coord[6][1] = 1; v_coord[6][2] = 1;
        v_coord[7][0] = 1; v_coord[7][1] = 1; v_coord[7][2] = 1;

        if (num_fem == 6)
        {
            t_map[0][0] = 0; t_map[0][1] = 2; t_map[0][2] = 1; t_map[0][3] = 5;
            t_map[1][0] = 1; t_map[1][1] = 2; t_map[1][2] = 3; t_map[1][3] = 5;
            t_map[2][0] = 0; t_map[2][1] = 4; t_map[2][2] = 2; t_map[2][3] = 5;
            t_map[3][0] = 5; t_map[3][1] = 3; t_map[3][2] = 7; t_map[3][3] = 2;
            t_map[4][0] = 4; t_map[4][1] = 5; t_map[4][2] = 6; t_map[4][3] = 2;
            t_map[5][0] = 5; t_map[5][1] = 7; t_map[5][2] = 6; t_map[5][3] = 2;
        }
        else if (num_fem == 8)
        {
            t_map[0][0] = 0; t_map[0][1] = 2; t_map[0][2] = 1; t_map[0][3] = 4;
            t_map[1][0] = 1; t_map[1][1] = 0; t_map[1][2] = 3; t_map[1][3] = 5;
            t_map[2][0] = 2; t_map[2][1] = 6; t_map[2][2] = 3; t_map[2][3] = 0;
            t_map[3][0] = 3; t_map[3][1] = 2; t_map[3][2] = 7; t_map[3][3] = 1;
            t_map[4][0] = 4; t_map[4][1] = 5; t_map[4][2] = 6; t_map[4][3] = 0;
            t_map[5][0] = 5; t_map[5][1] = 7; t_map[5][2] = 4; t_map[5][3] = 1;
            t_map[6][0] = 6; t_map[6][1] = 7; t_map[6][2] = 2; t_map[6][3] = 4;
            t_map[7][0] = 7; t_map[7][1] = 3; t_map[7][2] = 6; t_map[7][3] = 5;
        }
        else
        {
            printf("Wrong number of tetrahedrals\n");
            exit(EXIT_SUCCESS);
        }
    }
}

void x_map(double *&x, double *r, double **x_t, int n_dim, vector<function<double (double*)>> phi)
{
    for (int d = 0; d < n_dim; d++)
    {
        x[d] = 0.0;

        for (int i = 0; i < n_dim + 1; i++)
        {
            x[d] += x_t[d][i] * phi[i](r);
        }
    }
}

void J_xr_map(double **&J_xr, double *r, double **x_t, int n_dim, vector<function<void (double*, double*)>> dphi)
{
    double deriv[n_dim];

    for (int i = 0; i < n_dim; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            J_xr[i][j] = 0.0;

            for (int k = 0; k < n_dim + 1; k++)
            {
                dphi[k](deriv, r);

                J_xr[i][j] += x_t[i][k] * deriv[j];
            }
        }
    }
}

void parallel_ranking(long *&ranking, long *ids, int num_ids, long max_global)
{
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define how many elements are to sent to each processor
    int ids_count[size] = { 0 };

    for (int i = 0; i < num_ids; i++)
    {
        if (ids[i] < max_global)
        {
            ids_count[ids[i] / ((max_global + (size - 1)) / size)]++;
        }
    }

    // Notify each processor of how many items they will receive
    int receive_count[size];
    int total_receive = 0;

    MPI_Alltoall(ids_count, 1, MPI_INT, receive_count, 1, MPI_INT, MPI_COMM_WORLD);

    for (int p = 0; p < size; p++)
    {
        total_receive += receive_count[p];
    }

    // Create matrix with values corresponding to each processor
    long **ranking_values = new long*[size];
    int values_count[size] = { 0 };
    int **values_position = new int*[size];

    for (int p = 0; p < size; p++)
    {
        ranking_values[p] = new long[ids_count[p]];
        values_position[p] = new int[ids_count[p]];
    }

    for (int i = 0; i < num_ids; i++)
    {
        if (ids[i] < max_global)
        {
            int p = ids[i] / ((max_global + (size - 1)) / size);

            ranking_values[p][values_count[p]] = ids[i];
            values_position[p][values_count[p]] = i;
            values_count[p]++;
        }
    }

    // Map matrix of values of processors into 1D array
    int total_values = 0;
    int idx = 0;

    for (int p = 0; p < size; p++)
    {
        total_values += ids_count[p];
    }

    long message_values[total_values];

    for (int p = 0; p < size; p++)
    {
        for (int i = 0; i < ids_count[p]; i++)
        {
            message_values[idx] = ranking_values[p][i];
            idx++;
        }
    }

    // Comput offsets
    int message_offsets[size] = { 0 };
    int offset = 0;

    for (int p = 0; p < size; p++)
    {
        message_offsets[p] = offset;
        offset += ids_count[p];
    }

    long bucket_values[total_receive];
    int receive_offsets[size] = { 0 };
    offset = 0;

    for (int p = 0; p < size; p++)
    {
        receive_offsets[p] = offset;
        offset += receive_count[p];
    }

    // Send values to each processor to perform local rankings
    MPI_Alltoallv(message_values, ids_count, message_offsets, MPI_LONG, bucket_values, receive_count, receive_offsets, MPI_LONG, MPI_COMM_WORLD);

    // Perform local sorting
    vector<VertexID> local_data(total_receive);

    for (int i = 0; i < total_receive; i++)
    {
        local_data[i].key = i;
        local_data[i].value = bucket_values[i];
    }

    // Sort with respect to value
    sort(local_data.begin(), local_data.end(), [] (const VertexID &i, const VertexID &j) { return i.value < j.value; });

    // Set local ranking
    long ranking_value = 1;
    local_data[0].ranking = 0;

    for (int i = 1; i < total_receive; i++)
    {
        if (local_data[i].value == local_data[i - 1].value)
        {
            local_data[i].ranking = ranking_value - 1;
        }
        else
        {
            local_data[i].ranking = ranking_value;
            ranking_value++;
        }
    }

    // Share neighbor values to update ranking in each processor
    long left_value;

    if (rank < size - 1)
    {
        MPI_Send(&local_data[total_receive - 1].value, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0)
    {
        MPI_Recv(&left_value, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD, NULL);
    }

    // Perform scan to get rank offsets
    int ranking_offset = 0;
    long last_ranking = local_data[total_receive - 1].ranking;

    if ((rank > 0) && (left_value != local_data[0].value))
    {
        last_ranking++;
    }

    MPI_Scan(&last_ranking, &ranking_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    ranking_offset -= local_data[total_receive - 1].ranking;

    if (rank > 0)
    {
        for (int i = 0; i < total_receive; i++)
        {
            local_data[i].ranking += ranking_offset;
        }
    }

    // Sort with respect to key to get everything back in place
    sort(local_data.begin(), local_data.end(), [] (const VertexID &i, const VertexID &j) { return i.key < j.key; });

    // Send data to the corresponding processors
    for (int i = 0; i < total_receive; i++)
    {
        bucket_values[i] = local_data[i].ranking;
    }

    MPI_Alltoallv(bucket_values, receive_count, receive_offsets, MPI_LONG, message_values, ids_count, message_offsets, MPI_LONG, MPI_COMM_WORLD);

    // Reorganize data in original form
    idx = 0;

    for (int i = 0; i < num_ids; i++)
    {
        ranking[i] = -1;
    }

    for (int p = 0; p < size; p++)
    {
        for (int i = 0; i < ids_count[p]; i++)
        {
            ranking[values_position[p][i]] = message_values[idx];
            idx++;
        }
    }

    // Free memory
    for (int p = 0; p < size; p++)
    {
        delete[] ranking_values[p];
        delete[] values_position[p];
    }

    delete[] ranking_values;
    delete[] values_position;
}

void serial_ranking(long *ranking, long *array, int num_elems)
{
    vector<pair<int, long>> array_sorting(num_elems);

    for (int i = 0; i < num_elems; i++)
    {
        array_sorting[i].first = i;
        array_sorting[i].second = array[i];
    }

    sort(array_sorting.begin(), array_sorting.end(), [] (const pair<int, long> &i, const pair<int, long> &j) { return i.second < j.second; });

    long ranking_value = 1;
    ranking[0] = 0;

    for (int i = 1; i < num_elems; i++)
    {
        if (array_sorting[i].second == array_sorting[i - 1].second)
        {
            ranking[i] = ranking_value - 1;
        }
        else
        {
            ranking[i] = ranking_value;
            ranking_value++;
        }
    }

    for (int i = 0; i < num_elems; i++)
    {
        array_sorting[i].second = ranking[i];
    }

    sort(array_sorting.begin(), array_sorting.end(), [] (const pair<int, long> &i, const pair<int, long> &j) { return i.first < j.first; });

    for (int i = 0; i < num_elems; i++)
    {
        ranking[i] = array_sorting[i].second;
    }
}

// Math Functions
double determinant(double** A, int n)
{
    /*
     * Computes the determinant of a matrix
     */
    if (n == 2)
    {
        double d_1 = A[0][0] * A[1][1] - A[1][0] * A[0][1];

        return d_1;
    }
    else if (n == 3)
    {
        double d_1 = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
        double d_2 = A[0][1] * (A[1][0] * A[2][2] - A[2][0] * A[1][2]);
        double d_3 = A[0][2] * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);

        return d_1 - d_2 + d_3;
    }
    else if (n == 3)
    {
        double d_1 = A[0][0];
        double d_11 = A[1][1] * (A[2][2] * A[3][3] - A[3][2] * A[2][3]);
        double d_12 = A[1][2] * (A[2][1] * A[3][3] - A[3][1] * A[2][3]);
        double d_13 = A[1][3] * (A[2][1] * A[3][2] - A[3][1] * A[2][2]);

        double d_2 = A[0][1];
        double d_21 = A[1][0] * (A[2][2] * A[3][3] - A[3][2] * A[2][3]);
        double d_22 = A[1][2] * (A[2][0] * A[3][3] - A[3][0] * A[2][3]);
        double d_23 = A[1][3] * (A[2][0] * A[3][2] - A[3][0] * A[2][2]);

        double d_3 = A[0][2];
        double d_31 = A[1][0] * (A[2][1] * A[3][3] - A[3][1] * A[2][3]);
        double d_32 = A[1][1] * (A[2][0] * A[3][3] - A[3][0] * A[2][3]);
        double d_33 = A[1][3] * (A[2][0] * A[3][1] - A[3][0] * A[2][1]);

        double d_4 = A[0][3];
        double d_41 = A[1][0] * (A[2][1] * A[3][2] - A[3][1] * A[2][2]);
        double d_42 = A[1][1] * (A[2][0] * A[3][2] - A[3][0] * A[2][2]);
        double d_43 = A[1][2] * (A[2][0] * A[3][1] - A[3][0] * A[2][1]);

        return d_1 * (d_11 - d_12 + d_13) - d_2 * (d_21 - d_22 + d_23) + d_3 * (d_31 - d_32 + d_33) - d_4 * (d_41 - d_42 + d_43);
    }
    else
    {
        exit(EXIT_FAILURE);
    }
}

void inverse(double**& inv_A, double** A, int n)
{
    /*
     * Computes the inverse of a matrix
     */
    double det_A = determinant(A, n);

    if (n == 2)
    {
        inv_A[0][0] = (1.0 / det_A) * A[1][1];
        inv_A[0][1] = -(1.0 / det_A) * A[0][1];
        inv_A[1][0] = -(1.0 / det_A) * A[1][0];
        inv_A[1][1] = (1.0 / det_A) * A[0][0];
    }
    else if (n == 3)
    {
        inv_A[0][0] = (1.0 / det_A) * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
        inv_A[0][1] = (1.0 / det_A) * (A[0][2] * A[2][1] - A[2][2] * A[0][1]);
        inv_A[0][2] = (1.0 / det_A) * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
        inv_A[1][0] = (1.0 / det_A) * (A[1][2] * A[2][0] - A[2][2] * A[1][0]);
        inv_A[1][1] = (1.0 / det_A) * (A[0][0] * A[2][2] - A[2][0] * A[0][2]);
        inv_A[1][2] = (1.0 / det_A) * (A[0][2] * A[1][0] - A[1][2] * A[0][0]);
        inv_A[2][0] = (1.0 / det_A) * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);
        inv_A[2][1] = (1.0 / det_A) * (A[0][1] * A[2][0] - A[2][1] * A[0][0]);
        inv_A[2][2] = (1.0 / det_A) * (A[0][0] * A[1][1] - A[1][0] * A[0][1]);
    }
    else
    {
        exit(EXIT_FAILURE);
    }
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
