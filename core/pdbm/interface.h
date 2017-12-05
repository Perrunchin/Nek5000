/*
 * Declaration of interface functions to connect Fortran with C/C++
 */
// Headers

// Declarations
#ifndef INTERFACE_H
#define INTERFACE_H

// Global variables ceclaration
extern int n_x, n_y, n_z, n_elem, n_dim;
extern double ***mesh;
extern long **glo_num;
extern double **press_mask;

// Functions declaration
extern "C"
{
    // Set functions
    void enable_mpi_output_();
    void set_element_data_(int&, int&, int&, int&, int&);
    void set_mesh_data_(double*, double*, double*);
    void set_global_numbering_(long*);
    void set_pressure_mask_(double*);
}

// Memory management
template<typename DataType>
DataType* mem_alloc(int);

template<typename DataType>
DataType** mem_alloc(int, int);

template<typename DataType>
DataType*** mem_alloc(int, int, int);

template<typename DataType>
void mem_free(DataType*, int);

template<typename DataType>
void mem_free(DataType**, int, int);

template<typename DataType>
void mem_free(DataType***, int, int, int);

#endif
