/*
 * FEM functions definition
 */
// Declaration Headers
#include "fem.h"
#include "interface.h"

// C/C++ Headers
#include <iostream>
#include <functional>
#include <fstream>
#include <algorithm>
#include <cmath>

// Hypre Headers
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// Global variables definition
HYPRE_IJMatrix A_bc;
HYPRE_ParCSRMatrix A_fem;
HYPRE_IJMatrix B_bc;
HYPRE_ParCSRMatrix B_fem;
HYPRE_IJVector f_bc;
HYPRE_ParVector f_fem;
HYPRE_IJVector u_bc;
HYPRE_ParVector u_fem;
HYPRE_IJVector Bf_bc;
HYPRE_ParVector Bf_fem;
double *Binv_sem;
double *Bd_fem;
int num_nodes;
double *gll_nodes;
std::vector<std::function<double (double, double)>> phi;
std::vector<std::function<void (double *, double,  double)>> dphi;

// Functions definition
void assemble_fem_matrices_()
{
    // Compute mesh connectivity of the SEM matrix
    int num_sub_elem;
    int num_grid_points;
    double** V_sem;
    long int** E_sem;

    mesh_connectivity(V_sem, E_sem, num_grid_points, num_sub_elem);

    // Compute mesh connectivity of the FEM matrix
    int num_fem_elem;
    double** V_fem = V_sem;
    long int** E_fem;

    if (n_dim == 2)
    {
        rectangular_to_triangular(E_fem, num_fem_elem, E_sem, num_sub_elem);
    }
    else
    {
        hexahedral_to_tetrahedral(E_fem, num_fem_elem, E_sem, num_sub_elem);
    }

    // Basis functions
    generate_basis();

    // Generate FEM Matrix
    fem_matrices(V_fem, E_fem, num_fem_elem);

    // Vectors
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, max_rank - 1, &u_bc);
    HYPRE_IJVectorSetObjectType(u_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(u_bc);
    HYPRE_IJVectorAssemble(u_bc);
    HYPRE_IJVectorGetObject(u_bc, (void**) &u_fem);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, max_rank - 1, &f_bc);
    HYPRE_IJVectorSetObjectType(f_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(f_bc);
    HYPRE_IJVectorAssemble(f_bc);
    HYPRE_IJVectorGetObject(f_bc, (void**) &f_fem);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, max_rank - 1, &Bf_bc);
    HYPRE_IJVectorSetObjectType(Bf_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Bf_bc);
    HYPRE_IJVectorAssemble(Bf_bc);
    HYPRE_IJVectorGetObject(Bf_bc, (void**) &Bf_fem);

    // Free memory
    free_double_pointer(V_sem, n_x * n_y * n_z * n_elem);
    free_double_pointer(E_sem, num_sub_elem);
    free_double_pointer(E_fem, num_fem_elem);
}

void set_sem_inverse_mass_matrix_(double* inv_B)
{
    /*
     * Create Epetra_Vector for inv_B_sem
     */
    Binv_sem = new double[n_x * n_y * n_z * n_elem];

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        Binv_sem[i] = inv_B[i];
    }
}

void mesh_connectivity(double**& V_sem, long int**& E_sem, int& num_grid_points, int& num_sub_elem)
{
    /*
     * Computes the mesh connectivity of the SEM mesh including GLL points
     */
    // Vertices allocation
    num_grid_points = n_x * n_y * n_z * n_elem;

    V_sem = new double*[num_grid_points];

    for (int i = 0; i < num_grid_points; i++)
    {
        V_sem[i] = new double[n_dim];
    }

    // Elements allocation
    if (n_dim == 2)
    {
        num_sub_elem = (n_x - 1) * (n_y - 1) * n_elem;
    }
    else
    {
        num_sub_elem = (n_x - 1) * (n_y - 1) * (n_z - 1) * n_elem;
    }

    int num_vertices = std::pow(2, n_dim);

    E_sem = new long int*[num_sub_elem];

    for (int i = 0; i < num_sub_elem; i++)
    {
        E_sem[i] = new long int[num_vertices];
    }

    // Vertices generation
    for (int e = 0; e < n_elem; e++)
    {
        for (int ijk = 0; ijk < n_x * n_y * n_z; ijk++)
        {
            int grid_point = ijk + e * (n_x * n_y * n_z);

            if (n_dim == 2)
            {
                V_sem[grid_point][0] = x_m[grid_point];
                V_sem[grid_point][1] = y_m[grid_point];
            }
            else
            {
                V_sem[grid_point][0] = x_m[grid_point];
                V_sem[grid_point][1] = y_m[grid_point];
                V_sem[grid_point][2] = z_m[grid_point];
            }
        }
    }

    // Elements generation
    num_sub_elem = 0;

    for (int e = 0; e < n_elem; e++)
    {
        int elem = e * (n_x * n_y * n_z);

        if (n_dim == 2)
        {
            for (int j = 0; j < n_y - 1; j++)
            {
                for (int i = 0; i < n_x - 1; i++)
                {
                    int v_1 = i + j * n_x + elem;
                    int v_2 = (i + 1) + j * n_x + elem;
                    int v_3 = i + (j + 1) * n_x + elem;
                    int v_4 = (i + 1) + (j + 1) * n_x + elem;

                    E_sem[num_sub_elem][0] = v_1;
                    E_sem[num_sub_elem][1] = v_2;
                    E_sem[num_sub_elem][2] = v_3;
                    E_sem[num_sub_elem][3] = v_4;

                    num_sub_elem++;
                }
            }
        }
        else
        {
            for (int k = 0; k < n_z - 1; k++)
            {
                for (int j = 0; j < n_y - 1; j++)
                {
                    for (int i = 0; i < n_x - 1; i++)
                    {
                        int v_1 = i + j * n_x + k * (n_x * n_y) + elem;
                        int v_2 = (i + 1) + j * n_x + k * (n_x * n_y) + elem;
                        int v_3 = i + (j + 1) * n_x + k * (n_x * n_y) + elem;
                        int v_4 = (i + 1) + (j + 1) * n_x + k * (n_x * n_y) + elem;
                        int v_5 = i + j * n_x + (k + 1) * (n_x * n_y) + elem;
                        int v_6 = (i + 1) + j * n_x + (k + 1) * (n_x * n_y) + elem;
                        int v_7 = i + (j + 1) * n_x + (k + 1) * (n_x * n_y) + elem;
                        int v_8 = (i + 1) + (j + 1) * n_x + (k + 1) * (n_x * n_y) + elem;

                        E_sem[num_sub_elem][0] = v_1;
                        E_sem[num_sub_elem][1] = v_2;
                        E_sem[num_sub_elem][2] = v_3;
                        E_sem[num_sub_elem][3] = v_4;
                        E_sem[num_sub_elem][4] = v_5;
                        E_sem[num_sub_elem][5] = v_6;
                        E_sem[num_sub_elem][6] = v_7;
                        E_sem[num_sub_elem][7] = v_8;

                        num_sub_elem++;
                    }
                }
            }
        }
    }
}

void rectangular_to_triangular(long int**& E_fem, int& num_fem_elem, long int** E_sem, int num_sub_elem)
{
    /*
     * Computes the triangular elements from a rectangular mesh
     */
    // Allocate element array
    const int tri_per_elem = 2;
    //const int tri_per_elem = 4;

    num_fem_elem = tri_per_elem * num_sub_elem;

    E_fem = new long int*[num_fem_elem];

    for (int i = 0; i < num_fem_elem; i++)
    {
        E_fem[i] = new long int[3];
    }

    // Generate triangles
    int mapping[tri_per_elem][3] = {{0, 1, 3}, {0, 3, 2}};
    //int mapping[tri_per_elem][3] = {{0, 1, 2}, {1, 3, 0}, {2, 0, 3}, {3, 2, 1}};

    for (int e = 0; e < num_sub_elem; e++)
    {
        for (int i = 0; i < tri_per_elem; i++)
        {
            E_fem[tri_per_elem * e + i][0] = E_sem[e][mapping[i][0]];
            E_fem[tri_per_elem * e + i][1] = E_sem[e][mapping[i][1]];
            E_fem[tri_per_elem * e + i][2] = E_sem[e][mapping[i][2]];
        }
    }
}

void hexahedral_to_tetrahedral(long int**& E_fem, int& num_fem_elem, long int** E_sem, int num_sub_elem)
{
    /*
     * Computes the tetrahedral elements from a hexahedral mesh
     */
    // Allocate element array
    //const int tet_per_elem = 6;
    const int tet_per_elem = 8;
    //const int tet_per_elem = 16;

    //int mapping[tet_per_elem][n_dim + 1] = {{0, 2, 1, 5}, {1, 2, 3, 5}, {0, 4, 2, 5}, {5, 3, 7, 2}, {4, 5, 6, 2}, {5, 7, 6, 2}};
    int mapping[tet_per_elem][n_dim + 1] = {{0, 2, 1, 4}, {1, 0, 3, 5}, {2, 6, 3, 0}, {3, 2, 7, 1}, {4, 5, 6, 0}, {5, 7, 4, 1}, {6, 7, 2, 4}, {7, 3, 6, 5}};
    //int mapping[tet_per_elem][n_dim + 1] = {{0, 2, 1, 4}, {1, 0, 3, 5}, {2, 6, 3, 0}, {3, 2, 7, 1}, {4, 5, 6, 0}, {5, 7, 4, 1}, {6, 7, 2, 4}, {7, 3, 6, 5}, {0, 2, 1, 7}, {1, 0, 3, 6}, {2, 6, 3, 5}, {3, 2, 7, 4}, {4, 5, 6, 3}, {5, 7, 4, 2}, {6, 7, 2, 1}, {7, 3, 6, 0}};

    num_fem_elem = tet_per_elem * num_sub_elem;

    E_fem = new long int*[num_fem_elem];

    for (int i = 0; i < num_fem_elem; i++)
    {
        E_fem[i] = new long int[4];
    }

    // Generate tetrahedrals
    for (int e = 0; e < num_sub_elem; e++)
    {
        for (int i = 0; i < tet_per_elem; i++)
        {
            E_fem[tet_per_elem * e + i][0] = E_sem[e][mapping[i][0]];
            E_fem[tet_per_elem * e + i][1] = E_sem[e][mapping[i][1]];
            E_fem[tet_per_elem * e + i][2] = E_sem[e][mapping[i][2]];
            E_fem[tet_per_elem * e + i][3] = E_sem[e][mapping[i][3]];
        }
    }
}

// FEM Assembly
void generate_basis()
{
    // GLL Nodes
    num_nodes = n_x;
    double **P = allocate_double_pointer<double>(num_nodes, num_nodes);

    gll_nodes = new double[num_nodes];

    for (int i = 0; i < num_nodes; i++)
    {
        gll_nodes[i] = cos(M_PI * (double)(i) / (double)(num_nodes - 1));
    }

    double *gll_nodes_temp = new double[num_nodes];

    while (distance(gll_nodes, gll_nodes_temp, num_nodes) > std::numeric_limits<double>::epsilon())
    {
        for (int i = 0; i < num_nodes; i++)
        {
            gll_nodes_temp[i] = gll_nodes[i];
        }

        for (int i = 0; i < num_nodes; i++)
        {
            P[i][0] = 1.0;
            P[i][1] = gll_nodes[i];
        }

        for (int k = 1; k < num_nodes - 1; k++)
        {
            for (int i = 0; i < num_nodes; i++)
            {
                P[i][k + 1] = ((2.0 * k + 1.0) * gll_nodes[i] * P[i][k] - k * P[i][k - 1]) / (k + 1.0);
            }
        }

        for (int i = 0; i < num_nodes; i++)
        {
            gll_nodes[i] = gll_nodes_temp[i] - (gll_nodes[i] * P[i][num_nodes - 1] - P[i][num_nodes - 2]) / (num_nodes * P[i][num_nodes - 1]);
        }
    }

    for (int i = 0; i < num_nodes; i++)
    {
        gll_nodes_temp[i] = gll_nodes[num_nodes - 1 - i];
    }

    for (int i = 0; i < num_nodes; i++)
    {
        gll_nodes[i] = gll_nodes_temp[i];
    }

    free_single_pointer(gll_nodes_temp);
    free_double_pointer(P, num_nodes);

    // Functions
    phi.push_back([] (double r, double s) { return r; });
    phi.push_back([] (double r, double s) { return s; });
    phi.push_back([] (double r, double s) { return 1.0 - r - s; });

    dphi.push_back([] (double *dp, double r, double s) { dp[0] = 1.0; dp[1] = 0.0; });
    dphi.push_back([] (double *dp, double r, double s) { dp[0] = 0.0; dp[1] = 1.0; });
    dphi.push_back([] (double *dp, double r, double s) { dp[0] = -1.0; dp[1] = -1.0; });
}

void fem_matrices(double **V, long int **E, int num_elements)
{
    /*
     * Assembles the FEM matrices from (V, E)
     *
     * Returns A_fem and B_fem
     */
    // Create full matrix
    int num_vertices = *std::max_element(glo_num, glo_num + n_x * n_y * n_z * n_elem) + 1;

    HYPRE_IJMatrix A_full;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_vertices - 1, 0, num_vertices - 1, &A_full);
    HYPRE_IJMatrixSetObjectType(A_full, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_full);

    HYPRE_IJMatrix B_full;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_vertices - 1, 0, num_vertices - 1, &B_full);
    HYPRE_IJMatrixSetObjectType(B_full, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(B_full);

    // FEM Variables
    double **A_loc = allocate_double_pointer<double>(n_dim + 1, n_dim + 1);
    double **B_loc = allocate_double_pointer<double>(n_dim + 1, n_dim + 1);
    double **J_pr = allocate_double_pointer<double>(n_dim, n_dim);
    double **J_rp = allocate_double_pointer<double>(n_dim, n_dim);
    double **J_xp = allocate_double_pointer<double>(n_dim, n_dim);
    double **J_px = allocate_double_pointer<double>(n_dim, n_dim);
    double **J_xr = allocate_double_pointer<double>(n_dim, n_dim);

    // Quadrature rule
    int n_quad;
    double* q_r;
    double* q_s;
    double* q_t;
    double* q_w;

    if (n_dim == 2)
    {
        n_quad = 3;
        q_r = new double[n_quad];
        q_s = new double[n_quad];
        q_t = new double[n_quad];
        q_w = new double[n_quad];

        q_r[0] = 1.0 / 6.0;
        q_r[1] = 2.0 / 3.0;
        q_r[2] = 1.0 / 6.0;

        q_s[0] = 1.0 / 6.0;
        q_s[1] = 1.0 / 6.0;
        q_s[2] = 2.0 / 3.0;

        q_t[0] = 0.0;
        q_t[1] = 0.0;
        q_t[2] = 0.0;

        q_w[0] = 1.0 / 6.0;
        q_w[1] = 1.0 / 6.0;
        q_w[2] = 1.0 / 6.0;
    }
    else
    {
        n_quad = 4;
        q_r = new double[n_quad];
        q_s = new double[n_quad];
        q_t = new double[n_quad];;
        q_w = new double[n_quad];

        double a = (5.0 + 3.0 * sqrt(5.0)) / 20.0;
        double b = (5.0 - sqrt(5.0)) / 20.0;

        q_r[0] = a;
        q_r[1] = b;
        q_r[2] = b;
        q_r[3] = b;

        q_s[0] = b;
        q_s[1] = a;
        q_s[2] = b;
        q_s[3] = b;

        q_t[0] = b;
        q_t[1] = b;
        q_t[2] = a;
        q_t[3] = b;

        q_w[0] = 1.0 / 24.0;
        q_w[1] = 1.0 / 24.0;
        q_w[2] = 1.0 / 24.0;
        q_w[3] = 1.0 / 24.0;
    }

    // FEM Assembly process
    int E_x = num_nodes - 1;
    int E_y = num_nodes - 1;
    int num_tri = 2;
    int ***t_map = new int**[num_tri];

    for (int t = 0; t < num_tri; t++)
    {
        t_map[t] = new int*[n_dim + 1];

        for (int idx = 0; idx < n_dim + 1; idx++)
        {
            t_map[t][idx] = new int[n_dim];
        }
    }

    t_map[0][0][0] = 0;
    t_map[0][0][1] = 0;
    t_map[0][1][0] = 1;
    t_map[0][1][1] = 0;
    t_map[0][2][0] = 1;
    t_map[0][2][1] = 1;
    t_map[1][0][0] = 0;
    t_map[1][0][1] = 0;
    t_map[1][1][0] = 1;
    t_map[1][1][1] = 1;
    t_map[1][2][0] = 0;
    t_map[1][2][1] = 1;

    for (int e_y = 0; e_y < E_y; e_y++)
    {
        for (int e_x = 0; e_x < E_x; e_x++)
        {
            for (int t = 0; t < num_tri; t++)
            {
                // Reset local stiffness and mass matrices
                for (int i = 0; i < n_dim + 1; i++)
                {
                    for (int j = 0; j < n_dim + 1; j++)
                    {
                        A_loc[i][j] = 0.0;
                        B_loc[i][j] = 0.0;
                    }
                }

                // Get triangle
                double x[n_dim + 1], y[n_dim + 1];
                x[0] = gll_nodes[e_x + t_map[t][0][0]];
                y[0] = gll_nodes[e_y + t_map[t][0][1]];
                x[1] = gll_nodes[e_x + t_map[t][1][0]];
                y[1] = gll_nodes[e_y + t_map[t][1][1]];
                x[2] = gll_nodes[e_x + t_map[t][2][0]];
                y[2] = gll_nodes[e_y + t_map[t][2][1]];

                // Apply quadrature
                for (int q = 0; q < n_quad; q++)
                {
                    // From r to p
                    double q_p = p_map(q_r[q], q_s[q], x, y);
                    double q_q = q_map(q_r[q], q_s[q], x, y);
                    J_pr_map(J_pr, q_r[q], q_s[q], x, y);
                    inverse(J_rp, J_pr, n_dim);

                    // From p to x
                    J_xp_map(J_xp, q_p, q_q);
                    inverse(J_px, J_xp, n_dim);

                    // From r to x
                    for (int i = 0; i < n_dim; i++)
                    {
                        for (int j = 0; j < n_dim; j++)
                        {
                            J_xr[i][j] = 0.0;

                            for (int k = 0; k < n_dim; k++)
                            {
                                J_xr[i][j] += J_xp[i][k] * J_pr[k][j];
                            }
                        }
                    }

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
                                    double c = 0.0, d = 0.0;

                                    for (int gamma = 0; gamma < n_dim; gamma++)
                                    {
                                        double dp[n_dim];

                                        dphi[i](dp, q_r[q], q_s[q]);
                                        c += dp[gamma] * J_rp[gamma][beta];

                                        dphi[j](dp, q_r[q], q_s[q]);
                                        d += dp[gamma] * J_rp[gamma][beta];
                                    }

                                    a += c * J_px[beta][alpha];
                                    b += d * J_px[beta][alpha];
                                }

                                func += a * b;
                            }

                            A_loc[i][j] += func * det_J_xr * q_w[q];
                            B_loc[i][j] += phi[i](q_r[q], q_s[q]) * phi[j](q_r[q], q_s[q]) * det_J_xr * q_w[q];
                        }
                    }
                }

                // Add to global matrix
                int elem_id[n_dim + 1];
                elem_id[0] = (e_x + t_map[t][0][0]) + (e_y + t_map[t][0][1]) * n_x;
                elem_id[1] = (e_x + t_map[t][1][0]) + (e_y + t_map[t][1][1]) * n_x;
                elem_id[2] = (e_x + t_map[t][2][0]) + (e_y + t_map[t][2][1]) * n_x;

                for (int i = 0; i < n_dim + 1; i++)
                {
                    for (int j = 0; j < n_dim + 1; j++)
                    {
                        int row = glo_num[elem_id[i]];
                        int col = glo_num[elem_id[j]];

                        double A_val = A_loc[i][j];
                        double B_val = B_loc[i][j];

                        int ncols = 1;
                        int insert_error;

                        if (std::abs(A_val) > 1.0e-14)
                        {
                            insert_error = HYPRE_IJMatrixAddToValues(A_full, 1, &ncols, &row, &col, &A_val);
                        }


                        if (std::abs(B_val) > 1.0e-14)
                        {
                            insert_error = HYPRE_IJMatrixAddToValues(B_full, 1, &ncols, &row, &col, &B_val);
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

    // Assemble full matrix
    HYPRE_IJMatrixAssemble(A_full);
    HYPRE_IJMatrixAssemble(B_full);

    // Get index of glo_num
    std::vector<std::pair<long int, long int>> glo_num_pair;

    for (int i = 0; i < n_x * n_y * n_z * n_elem; i++)
    {
        glo_num_pair.push_back(std::pair<long int, long int>(glo_num[i], i));
    }

    std::sort(glo_num_pair.begin(), glo_num_pair.end());

    int id = 1;
    long int* glo_num_index = new long int[num_vertices]; // Vertex "i" is in position "glo_num_index[i]" in "glo_num"

    glo_num_index[0] = glo_num_pair[0].second;

    for (int i = 1; i < n_x * n_y * n_z * n_elem; i++)
    {
        if (glo_num_pair[i - 1].first != glo_num_pair[i].first)
        {
            glo_num_index[id] = glo_num_pair[i].second;

            id++;
        }
    }

    // Extract CRS arrays and apply Dirichlet boundary conditions
    long int num_rows = max_rank;

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_rows - 1, 0, num_rows - 1, &A_bc);
    HYPRE_IJMatrixSetObjectType(A_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_bc);

    HYPRE_ParCSRMatrix A_full_csr;
    HYPRE_IJMatrixGetObject(A_full, (void**) &A_full_csr);

    for (int i = 0; i < num_vertices; i++)
    {
        int row = ranking[glo_num_index[i]];

        if (row < num_rows)
        {
            int ncols;
            int* cols;
            double* values;

            HYPRE_ParCSRMatrixGetRow(A_full_csr, i, &ncols, &cols, &values);

            for (int j = 0; j < ncols; j++)
            {
                int col = ranking[glo_num_index[cols[j]]];

                if (col < num_rows)
                {
                    int num_cols = 1;
                    int insert_error = HYPRE_IJMatrixAddToValues(A_bc, 1, &num_cols, &row, &col, values + j);

                    if (insert_error != 0)
                    {
                        printf("There was an error with entry A_fem(%d, %d) = %f\n", row, col, values[j]);
                        exit(EXIT_FAILURE);
                    }
                }
            }

            HYPRE_ParCSRMatrixRestoreRow(A_full_csr, i, &ncols, &cols, &values);
        }
    }

    HYPRE_IJMatrixAssemble(A_bc);

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_rows - 1, 0, num_rows - 1, &B_bc);
    HYPRE_IJMatrixSetObjectType(B_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(B_bc);

    HYPRE_ParCSRMatrix B_full_csr;
    HYPRE_IJMatrixGetObject(B_full, (void**) &B_full_csr);

    for (int i = 0; i < num_vertices; i++)
    {
        int row = ranking[glo_num_index[i]];

        if (row < num_rows)
        {
            int ncols;
            int* cols;
            double* values;

            HYPRE_ParCSRMatrixGetRow(B_full_csr, i, &ncols, &cols, &values);

            for (int j = 0; j < ncols; j++)
            {
                int col = ranking[glo_num_index[cols[j]]];

                if (col < num_rows)
                {
                    int num_cols = 1;
                    int insert_error = HYPRE_IJMatrixAddToValues(B_bc, 1, &num_cols, &row, &col, values + j);

                    if (insert_error != 0)
                    {
                        printf("There was an error with entry B_fem(%d, %d) = %f\n", row, col, values[j]);
                        exit(EXIT_FAILURE);
                    }
                }
            }

            HYPRE_ParCSRMatrixRestoreRow(B_full_csr, i, &ncols, &cols, &values);
        }
    }

    HYPRE_IJMatrixAssemble(B_bc);

    // Build diagonal mass matrix with full mass matrix without boundary conditions
    double* Bd_sum = new double[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        int ncols;
        int* cols;
        double* values;

        HYPRE_ParCSRMatrixGetRow(B_full_csr, i, &ncols, &cols, &values);

        Bd_sum[i] = 0.0;

        for (int j = 0; j < ncols; j++)
        {
            Bd_sum[i] += values[j];
        }

        HYPRE_ParCSRMatrixRestoreRow(B_full_csr, i, &ncols, &cols, &values);
    }

    Bd_fem = new double[num_rows];

    for (int i = 0; i < num_vertices; i++)
    {
        int row = ranking[glo_num_index[i]];

        if (row < num_rows)
        {
            Bd_fem[row] = Bd_sum[i];
        }
    }

    // Create ParCSR objects
    HYPRE_IJMatrixGetObject(A_bc, (void**) &A_fem);
    HYPRE_IJMatrixGetObject(B_bc, (void**) &B_fem);

    // Free memory
    HYPRE_IJMatrixDestroy(A_full);
    HYPRE_IJMatrixDestroy(B_full);
    delete[] glo_num_index;
    delete[] q_w;
    delete[] q_r;
    delete[] q_s;
    delete[] q_t;
    delete[] Bd_sum;
    free_double_pointer(A_loc, n_dim + 1);
    free_double_pointer(B_loc, n_dim + 1);
    free_double_pointer(J_rp, n_dim);
    free_double_pointer(J_pr, n_dim);
    free_double_pointer(J_xp, n_dim);
    free_double_pointer(J_px, n_dim);
    free_double_pointer(J_xr, n_dim);
}

// Geometric functions
double x_map(double p, double q)
{
    switch (mapping)
    {
        case 1:
            // Linear
            return (lambda + (1.0 / 2.0) * (1.0 - lambda) * (1.0 - q)) * p;

        case 2:
            // Parabola
            return ((1.0 - lambda) * pow(q, 2.0) + lambda) * p;

        case 3:
            // Cosine
            return ((1.0 / 2.0) * (lambda + 1.0 + (lambda - 1.0) * cos(M_PI * q))) * p;

        default:
            // None
            return p;
    }
}

double y_map(double p, double q)
{
    switch (mapping)
    {
        case 1:
            // Linear
            return q;

        case 2:
            // Parabola
            return q;

        case 3:
            // Cosine
            return q;

        default:
            // None
            return q;
    }
}

void J_xp_map(double **J_xp, double p, double q)
{
    double dx_dp, dx_dq, dy_dp, dy_dq;

    switch (mapping)
    {
        case 1:
            // Linear
            dx_dp = lambda + (1.0 / 2.0) * (1.0 - lambda) * (1.0 - q);
            dx_dq = (1.0 / 2.0) * (lambda - 1.0) * p;
            dy_dp = 0.0;
            dy_dq = 1.0;

            break;

        case 2:
            // Parabola
            dx_dp = (1.0 - lambda) * pow(q, 2.0) + lambda;
            dx_dq = 2.0 * p * q * (1.0 - lambda);
            dy_dp = 0.0;
            dy_dq = 1.0;

            break;

        case 3:
            // Cosine
            dx_dp = (1.0 / 2.0) * (lambda + 1.0 + (lambda - 1.0) * cos(M_PI * q));
            dx_dq = - (M_PI / 2.0) * p * (lambda - 1.0) * sin(M_PI * q);
            dy_dp = 0.0;
            dy_dq = 1.0;

            break;

        default:
            // None
            dx_dp = 1.0;
            dx_dq = 0.0;
            dy_dp = 0.0;
            dy_dq = 1.0;

            break;
    }

    J_xp[0][0] = dx_dp;
    J_xp[0][1] = dx_dq;
    J_xp[1][0] = dy_dp;
    J_xp[1][1] = dy_dq;
}

double p_map(double r, double s, double x[], double y[])
{
    double p = 0.0;

    for (int i = 0; i < n_dim + 1; i++)
    {
        p += x[i] * phi[i](r, s);
    }

    return p;
}

double q_map(double r, double s, double x[], double y[])
{
    double q = 0.0;

    for (int i = 0; i < n_dim + 1; i++)
    {
        q += y[i] * phi[i](r, s);
    }

    return q;
}

void J_pr_map(double **J_pr, double r, double s, double x[], double y[])
{
    double dp_dr = 0.0, dp_ds = 0.0, dq_dr = 0.0, dq_ds = 0.0;
    double deriv[n_dim];

    for (int i = 0; i < n_dim + 1; i++)
    {
        dphi[i](deriv, r, s);

        dp_dr += x[i] * deriv[0];
        dp_ds += x[i] * deriv[1];
        dq_dr += y[i] * deriv[0];
        dq_ds += y[i] * deriv[1];
    }

    J_pr[0][0] = dp_dr;
    J_pr[0][1] = dp_ds;
    J_pr[1][0] = dq_dr;
    J_pr[1][1] = dq_ds;
}

// Math Functions
double interp(double x_p, double *x, double *y, int size)
{
    /*
     * Evaluate the interpolating polynomial trough the data (x, y) at point x_p
     */
    double p_x = 0.0;

    for (int i = 0; i < size; i++)
    {
        double ell_i = 1.0;

        for (int j = 0; j < size; j++)
        {
            if (j == i)
            {
                continue;
            }

            ell_i *= (x_p - x[j]) / (x[i] - x[j]);
        }

        p_x += y[i] * ell_i;
    }

    return p_x;
}

double distance(double *x, double *y, int size)
{
    double norm = 0.0;

    for (int i = 0; i < size; i++)
    {
        norm = std::max(norm, std::abs(x[i] - y[i]));
    }

    return norm;
}

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

void inverse(double**& inv_A, double** A, int n, double det_A)
{
    /*
     * Computes the inverse of a matrix
     */
    if (det_A == 0.0)
    {
        det_A = determinant(A, n);
    }

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

void matrix_matrix_mul(double** C, double** A, double** B, int n, int m, int l, bool A_T, bool B_T)
{
    /*
     * Computes C = A * B. A is of size (n, l), B is of size (l, m), and C is of size (n, m)
     */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < l; k++)
            {
                if (A_T)
                {
                    if (B_T)
                    {
                        sum += A[k][i] * B[j][k];
                    }
                    else
                    {
                        sum += A[k][i] * B[k][j];
                    }
                }
                else
                {
                    if (B_T)
                    {
                        sum += A[i][k] * B[j][k];
                    }
                    else
                    {
                        sum += A[i][k] * B[k][j];
                    }
                }
            }

            C[i][j] = sum;
        }
    }
}

void matrix_scaling(double** A, double value, int n, int m)
{
    /*
     * Scale a matrix by a constant
     */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            A[i][j] *= value;
        }
    }
}

void row_scaling(double** B, double** A, double* w, int n, int m)
{
    /*
     * Scales the row of matrix A with values in w and stores it in B
     */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            B[i][j] = A[i][j] * w[i];
        }
    }
}

// Utility functions
template<typename DataType>
DataType* allocate_single_pointer(int n)
{
    DataType* pointer = new DataType[n];

    return pointer;
}

template<typename PointerType>
void free_single_pointer(PointerType& single_pointer)
{
    delete[] single_pointer;
}

template<typename DataType>
DataType** allocate_double_pointer(int n, int m)
{
    DataType** pointer = new DataType*[n];

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType[m];
    }

    return pointer;
}

template<typename DataType>
DataType** allocate_double_pointer(int n, int m, double value)
{
    DataType** pointer = new DataType*[n];

    for (int i = 0; i < n; i++)
    {
        pointer[i] = new DataType[m];

        for (int j = 0; j < m; j++)
        {
            pointer[i][j] = value;
        }
    }

    return pointer;
}

template<typename PointerType>
void free_double_pointer(PointerType& double_pointer, int size)
{
    for (int i = 0; i < size; i++)
    {
        delete[] double_pointer[i];
    }

    delete[] double_pointer;
}

void print_matrix(double** A, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%f ", A[i][j]);
        }

        printf("\n");
    }
}

void print_vertices(double** V, int num_vert, int num_dim)
{
    for (int i = 0; i < num_vert; i++)
    {
        if (num_dim == 2)
        {
            printf("%d: (%f, %f)\n", i + 1, V[i][0], V[i][1]);
        }
        else
        {
            printf("%d: (%f, %f, %f)\n", i + 1, V[i][0], V[i][1], V[i][2]);
        }
    }
}

void print_elements(long int** E, int num_elem, int num_grid_points)
{
    for (int e = 0; e < num_elem; e++)
    {
        printf("[ ");

        for (int i = 0; i < num_grid_points; i++)
        {
            printf("%ld ", E[e][i] + 1);
        }

        printf("]\n");
    }
}
