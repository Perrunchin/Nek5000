/*
 * Declaration of utility functions
 */
// Headers

#ifndef UTILITIES_H
#define UTILITIES_H

extern "C"
{
    // In-source breakpoint to be called from Fortran
    void breakpoint_();
}

long maximum_value(long*, int);
long maximum_value(long**, int, int);

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
