//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: myAlexNetGPU_terminate.cu
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

// Include Files
#include "myAlexNetGPU_terminate.h"
#include "myAlexNetGPU.h"
#include "myAlexNetGPU_data.h"
#include <cstdio>

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void myAlexNetGPU_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "ERR[%d] %s:%s\n", errCode, cudaGetErrorName(errCode),
            cudaGetErrorString(errCode));
    exit(errCode);
  }
  myAlexNetGPU_free();
  isInitialized_myAlexNetGPU = false;
}

//
// File trailer for myAlexNetGPU_terminate.cu
//
// [EOF]
//
