//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: myAlexNetGPU.h
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

#ifndef MYALEXNETGPU_H
#define MYALEXNETGPU_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern double myAlexNetGPU(const unsigned char I_data[], const int I_size[3]);

void myAlexNetGPU_free();

void myAlexNetGPU_init();

#endif
//
// File trailer for myAlexNetGPU.h
//
// [EOF]
//
