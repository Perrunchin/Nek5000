/*
 * Declaration of interface functions to connect Fortran with C/C++
 */
// Headers

// Declarations
#ifndef INTERFACE_H
#define INTERFACE_H

// Global variables ceclaration
extern int n_x, n_y, n_z, n_elem, n_dim;
extern double* x_m;
extern double* y_m;
extern double* z_m;
extern long int* glo_num;
extern double* press_mask;
extern long int* ranking;
extern long int max_rank;
extern double* rhs;
extern long int* rhs_index;
extern int* indices;
extern double lambda;
extern int mapping;

// Functions declaration
extern "C"
{
    // Set functions
    void set_element_data_(int&, int&, int&, int&, int&);
    void set_mesh_data_(double*, double*, double*);
    void set_global_numbering_(long int*);
    void set_pressure_mask_(double*);
    void compute_ranking_();
    void set_lambda_(double&);
    void set_mapping_(int&);
    void save_mesh_data_();

    // Memory management functions
    void free_global_memory();

    // Displaying functions
    void print_variables_();
}

// Utility functions
template<typename PointerType>
void free_single_pointer(PointerType&);

template<typename PointerType>
void free_double_pointer(PointerType&, int);

#endif

