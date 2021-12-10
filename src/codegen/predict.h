//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: predict.h
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

#ifndef PREDICT_H
#define PREDICT_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

class alexnet0_0;

// Function Declarations
namespace coder {
void DeepLearningNetwork_predict(alexnet0_0 *obj,
                                 const unsigned char varargin_1[154587],
                                 float varargout_1[1000]);

}

#endif
//
// File trailer for predict.h
//
// [EOF]
//
