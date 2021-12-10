//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: predict.cu
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

// Include Files
#include "predict.h"
#include "DeepLearningNetwork.h"
#include "myAlexNetGPU_internal_types.h"
#include "MWCudaDimUtility.hpp"

// Type Definitions
struct cell_wrap_12 {
  float f1[1000];
};

struct cell_wrap_9 {
  float f1[154587];
};

// Function Declarations
static __global__ void
DeepLearningNetwork_predict_kernel21(const unsigned char varargin_1[154587],
                                     unsigned char input[154587]);

static __global__ void
DeepLearningNetwork_predict_kernel22(const unsigned char input[154587],
                                     cell_wrap_9 inMiniBatchGroup[1]);

static __global__ void
DeepLearningNetwork_predict_kernel23(const cell_wrap_12 outMiniBatchGroup[1],
                                     float varargout_1[1000]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char varargin_1[154587]
//                unsigned char input[154587]
// Return Type  : void
//
static __global__
    __launch_bounds__(512, 1) void DeepLearningNetwork_predict_kernel21(
        const unsigned char varargin_1[154587], unsigned char input[154587])
{
  unsigned long threadId;
  int i;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  i = static_cast<int>(threadId);
  if (i < 154587) {
    input[i] = varargin_1[i];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char input[154587]
//                cell_wrap_9 inMiniBatchGroup[1]
// Return Type  : void
//
static __global__
    __launch_bounds__(512, 1) void DeepLearningNetwork_predict_kernel22(
        const unsigned char input[154587], cell_wrap_9 inMiniBatchGroup[1])
{
  unsigned long threadId;
  int i;
  int i1;
  int p;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  i = static_cast<int>(threadId % 227UL);
  threadId = (threadId - static_cast<unsigned long>(i)) / 227UL;
  i1 = static_cast<int>(threadId % 227UL);
  threadId = (threadId - static_cast<unsigned long>(i1)) / 227UL;
  p = static_cast<int>(threadId);
  if ((static_cast<int>((static_cast<int>(p < 3)) &&
                        (static_cast<int>(i1 < 227)))) &&
      (static_cast<int>(i < 227))) {
    inMiniBatchGroup[0].f1[(i + 227 * i1) + 51529 * p] =
        static_cast<float>(input[(i1 + 227 * i) + 51529 * p]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const cell_wrap_12 outMiniBatchGroup[1]
//                float varargout_1[1000]
// Return Type  : void
//
static __global__
    __launch_bounds__(512, 1) void DeepLearningNetwork_predict_kernel23(
        const cell_wrap_12 outMiniBatchGroup[1], float varargout_1[1000])
{
  unsigned long threadId;
  int i;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  i = static_cast<int>(threadId);
  if (i < 1000) {
    varargout_1[i] = outMiniBatchGroup[0].f1[i];
  }
}

//
// Arguments    : alexnet0_0 *obj
//                const unsigned char varargin_1[154587]
//                float varargout_1[1000]
// Return Type  : void
//
namespace coder {
void DeepLearningNetwork_predict(alexnet0_0 *obj,
                                 const unsigned char varargin_1[154587],
                                 float varargout_1[1000])
{
  cell_wrap_12(*gpu_outMiniBatchGroup)[1];
  cell_wrap_9(*gpu_inMiniBatchGroup)[1];
  float(*gpu_varargout_1)[1000];
  unsigned char(*gpu_input)[154587];
  unsigned char(*gpu_varargin_1)[154587];
  cudaMalloc(&gpu_varargout_1, 4000UL);
  cudaMalloc(&gpu_outMiniBatchGroup, 4000UL);
  cudaMalloc(&gpu_inMiniBatchGroup, 618348UL);
  cudaMalloc(&gpu_input, 154587UL);
  cudaMalloc(&gpu_varargin_1, 154587UL);
  cudaMemcpy(*gpu_varargin_1, varargin_1, 154587UL, cudaMemcpyHostToDevice);
  DeepLearningNetwork_predict_kernel21<<<dim3(302U, 1U, 1U),
                                         dim3(512U, 1U, 1U)>>>(*gpu_varargin_1,
                                                               *gpu_input);
  DeepLearningNetwork_predict_kernel22<<<dim3(302U, 1U, 1U),
                                         dim3(512U, 1U, 1U)>>>(
      *gpu_input, *gpu_inMiniBatchGroup);
  cudaMemcpy(obj->getInputDataPointer(0), (*gpu_inMiniBatchGroup)[0].f1,
             obj->getLayerOutputSize(0, 0), cudaMemcpyDeviceToDevice);
  obj->predict();
  cudaMemcpy((*gpu_outMiniBatchGroup)[0].f1, obj->getLayerOutput(18, 0),
             obj->getLayerOutputSize(18, 0), cudaMemcpyDeviceToDevice);
  DeepLearningNetwork_predict_kernel23<<<dim3(2U, 1U, 1U),
                                         dim3(512U, 1U, 1U)>>>(
      *gpu_outMiniBatchGroup, *gpu_varargout_1);
  cudaMemcpy(varargout_1, *gpu_varargout_1, 4000UL, cudaMemcpyDeviceToHost);
  cudaFree(*gpu_varargin_1);
  cudaFree(*gpu_input);
  cudaFree(*gpu_inMiniBatchGroup);
  cudaFree(*gpu_outMiniBatchGroup);
  cudaFree(*gpu_varargout_1);
}

} // namespace coder

//
// File trailer for predict.cu
//
// [EOF]
//
