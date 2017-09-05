/*
 * Program to add breakpoints inside source code
 */

// Headers

// Function declaration and definition
void breakpoint_(void)
{
    asm("int $3");
}
