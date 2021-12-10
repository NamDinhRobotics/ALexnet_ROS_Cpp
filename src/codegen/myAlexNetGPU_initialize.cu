//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: myAlexNetGPU_initialize.cu
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

// Include Files
#include "myAlexNetGPU_initialize.h"
#include "myAlexNetGPU.h"
#include "myAlexNetGPU_data.h"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void myAlexNetGPU_initialize()
{
  myAlexNetGPU_init();
  cudaGetLastError();
  isInitialized_myAlexNetGPU = true;
}

//
// File trailer for myAlexNetGPU_initialize.cu
//
// [EOF]
//
