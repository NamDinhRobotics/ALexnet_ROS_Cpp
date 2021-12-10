//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: myAlexNetGPU.cu
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

// Include Files
#include "myAlexNetGPU.h"
#include "DeepLearningNetwork.h"
#include "myAlexNetGPU_data.h"
#include "myAlexNetGPU_initialize.h"
#include "myAlexNetGPU_internal_types.h"
#include "predict.h"
#include "MWCudaDimUtility.hpp"
#include "MWLaunchParametersUtilities.hpp"
#include <cmath>

// Variable Definitions
static alexnet0_0 net;

static bool net_not_empty;

// Function Declarations
static __global__ void
myAlexNetGPU_kernel1(const unsigned char I_data[], const int I_size[3],
                     const int img_size[3], const int b_I_size,
                     const int in_rows, unsigned char img_data[2211840]);

static __global__ void myAlexNetGPU_kernel10(const int plast,
                                             double ipColIndices_data[4313]);

static __global__ void myAlexNetGPU_kernel11(const int plast,
                                             double colWeights_data[4313]);

static __global__ void myAlexNetGPU_kernel12(const double aux2_data[2048],
                                             const int in_cols,
                                             const double scale_idx_1,
                                             const double kwidthCol,
                                             double ipColIndices_data[4313],
                                             double colWeights_data[4313]);

static __global__ void myAlexNetGPU_kernel13(const double rowWeights_data[2951],
                                             double rowWeightsTotal[227]);

static __global__ void myAlexNetGPU_kernel14(const double rowWeights_data[2951],
                                             const int in_rows,
                                             double rowWeightsTotal[227]);

static __global__ void myAlexNetGPU_kernel15(const double colWeights_data[4313],
                                             double colWeightsTotal[227]);

static __global__ void myAlexNetGPU_kernel16(const double colWeights_data[4313],
                                             const int in_rows,
                                             double colWeightsTotal[227]);

static __global__ void myAlexNetGPU_kernel17(
    const double colWeightsTotal[227], const double colWeights_data[4313],
    const unsigned char img_data[2211840], const double ipColIndices_data[4313],
    const int img_size[3], const int partialResize_size[3],
    const double kwidthCol, const int plast,
    unsigned char partialResize_data[490320]);

static __global__ void myAlexNetGPU_kernel18(
    const double rowWeightsTotal[227], const double rowWeights_data[2951],
    const unsigned char partialResize_data[490320],
    const int partialResize_size[3], const double ipRowIndices_data[2951],
    const double kwidthRow, unsigned char out[154587]);

static __global__ void myAlexNetGPU_kernel19(
    const double rowWeightsTotal[227], const double rowWeights_data[2951],
    const unsigned char img_data[2211840], const int img_size[3],
    const double ipRowIndices_data[2951], const int partialResize_size[3],
    const double kwidthRow, const int plast,
    unsigned char partialResize_data[697344]);

static __global__ void myAlexNetGPU_kernel2(const unsigned char I_data[],
                                            const int I_size,
                                            unsigned char img_data[2211840]);

static __global__ void myAlexNetGPU_kernel20(
    const double colWeightsTotal[227], const double colWeights_data[4313],
    const unsigned char partialResize_data[697344],
    const int partialResize_size[3], const double ipColIndices_data[4313],
    const double kwidthCol, unsigned char out[154587]);

static __global__ void
myAlexNetGPU_kernel3(const unsigned char img_data[2211840],
                     const int img_size[3], const int b_img_size[3],
                     const int plast, const int in_rows,
                     unsigned char b_img_data[2211840]);

static __global__ void
myAlexNetGPU_kernel4(const unsigned char img_data[2211840], const int img_size,
                     unsigned char b_img_data[2211840]);

static __global__ void myAlexNetGPU_kernel5(const int in_rows, const int plast,
                                            double aux1_data[1440]);

static __global__ void myAlexNetGPU_kernel6(const int in_cols, const int plast,
                                            double aux2_data[2048]);

static __global__ void myAlexNetGPU_kernel7(const int plast,
                                            double ipRowIndices_data[2951]);

static __global__ void myAlexNetGPU_kernel8(const int plast,
                                            double rowWeights_data[2951]);

static __global__ void myAlexNetGPU_kernel9(const double aux1_data[1440],
                                            const int in_rows,
                                            const double scale_idx_0,
                                            const double kwidthRow,
                                            double ipRowIndices_data[2951],
                                            double rowWeights_data[2951]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char I_data[]
//                const int I_size[3]
//                const int img_size[3]
//                const int b_I_size
//                const int in_rows
//                unsigned char img_data[2211840]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel1(
    const unsigned char I_data[], const int I_size[3], const int img_size[3],
    const int b_I_size, const int in_rows, unsigned char img_data[2211840])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = 3UL * ((static_cast<unsigned long>(b_I_size) + 1UL) *
                   (static_cast<unsigned long>(in_rows - 1) + 1UL)) -
            1UL;
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    unsigned long tmpIndex;
    int colIndices;
    int k;
    int rowIndices;
    colIndices = static_cast<int>(idx % 3UL);
    tmpIndex = (idx - static_cast<unsigned long>(colIndices)) / 3UL;
    rowIndices = static_cast<int>(tmpIndex %
                                  (static_cast<unsigned long>(b_I_size) + 1UL));
    tmpIndex = (tmpIndex - static_cast<unsigned long>(rowIndices)) /
               (static_cast<unsigned long>(b_I_size) + 1UL);
    k = static_cast<int>(tmpIndex);
    img_data[((static_cast<int>(static_cast<short>(k + 1)) +
               img_size[0] *
                   (static_cast<int>(static_cast<short>(rowIndices + 1)) - 1)) +
              img_size[0] * img_size[1] * colIndices) -
             1] = I_data
        [(colIndices +
          3 * (static_cast<int>(static_cast<short>(rowIndices + 1)) - 1)) +
         3 * I_size[1] * (static_cast<int>(static_cast<short>(k + 1)) - 1)];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int plast
//                double ipColIndices_data[4313]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel10(
    const int plast, double ipColIndices_data[4313])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(227 * plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    ipColIndices_data[oldIdx] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int plast
//                double colWeights_data[4313]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel11(
    const int plast, double colWeights_data[4313])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(227 * plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    colWeights_data[oldIdx] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double aux2_data[2048]
//                const int in_cols
//                const double scale_idx_1
//                const double kwidthCol
//                double ipColIndices_data[4313]
//                double colWeights_data[4313]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel12(
    const double aux2_data[2048], const int in_cols, const double scale_idx_1,
    const double kwidthCol, double ipColIndices_data[4313],
    double colWeights_data[4313])
{
  unsigned long threadId;
  int colIdx;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  colIdx = static_cast<int>(threadId);
  if (colIdx < 227) {
    int rowIndices;
    rowIndices = static_cast<int>(ceil(kwidthCol));
    for (int k{0}; k < rowIndices; k++) {
      double absx;
      double absx2;
      double sumVal;
      int b_y;
      int colIndices;
      int l;
      sumVal = (static_cast<double>(colIdx) + 1.0) / scale_idx_1 +
               0.5 * (1.0 - 1.0 / scale_idx_1);
      colIndices = static_cast<int>(floor(sumVal - kwidthCol / 2.0));
      sumVal -= static_cast<double>(colIndices + k) + 1.0;
      if (scale_idx_1 < 1.0) {
        sumVal *= scale_idx_1;
      }
      absx = fabs(sumVal);
      absx2 = absx * absx;
      sumVal = pow(absx, 3.0);
      sumVal = ((1.5 * sumVal - 2.5 * absx2) + 1.0) *
                   static_cast<double>(absx <= 1.0) +
               (((-0.5 * sumVal + 2.5 * absx2) - 4.0 * absx) + 2.0) *
                   static_cast<double>((static_cast<int>(1.0 < absx)) &&
                                       (static_cast<int>(absx <= 2.0)));
      if (scale_idx_1 < 1.0) {
        colWeights_data[colIdx + 227 * k] = scale_idx_1 * sumVal;
      } else {
        colWeights_data[colIdx + 227 * k] = sumVal;
      }
      b_y = in_cols << 1;
      colIndices = (colIndices + k) + 1;
      l = colIndices - 1;
      if (b_y == 0) {
        if (colIndices - 1 == 0) {
          l = 0;
        }
      } else if (colIndices - 1 == 0) {
        l = 0;
      } else {
        l = static_cast<int>(fmod(static_cast<double>(colIndices) - 1.0,
                                  static_cast<double>(b_y)));
        if ((static_cast<int>(l != 0)) &&
            (static_cast<int>(colIndices - 1 < 0))) {
          l += b_y;
        }
      }
      ipColIndices_data[colIdx + 227 * k] = aux2_data[l];
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeights_data[2951]
//                double rowWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel13(
    const double rowWeights_data[2951], double rowWeightsTotal[227])
{
  unsigned long threadId;
  int colIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  colIndices = static_cast<int>(threadId);
  if (colIndices < 227) {
    rowWeightsTotal[colIndices] = rowWeights_data[colIndices];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeights_data[2951]
//                const int in_rows
//                double rowWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel14(
    const double rowWeights_data[2951], const int in_rows,
    double rowWeightsTotal[227])
{
  unsigned long threadId;
  int colIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  colIndices = static_cast<int>(threadId);
  if (colIndices < 227) {
    rowWeightsTotal[colIndices] += rowWeights_data[in_rows + colIndices];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeights_data[4313]
//                double colWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel15(
    const double colWeights_data[4313], double colWeightsTotal[227])
{
  unsigned long threadId;
  int colIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  colIndices = static_cast<int>(threadId);
  if (colIndices < 227) {
    colWeightsTotal[colIndices] = colWeights_data[colIndices];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeights_data[4313]
//                const int in_rows
//                double colWeightsTotal[227]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel16(
    const double colWeights_data[4313], const int in_rows,
    double colWeightsTotal[227])
{
  unsigned long threadId;
  int colIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  colIndices = static_cast<int>(threadId);
  if (colIndices < 227) {
    colWeightsTotal[colIndices] += colWeights_data[in_rows + colIndices];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeightsTotal[227]
//                const double colWeights_data[4313]
//                const unsigned char img_data[2211840]
//                const double ipColIndices_data[4313]
//                const int img_size[3]
//                const int partialResize_size[3]
//                const double kwidthCol
//                const int plast
//                unsigned char partialResize_data[490320]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel17(
    const double colWeightsTotal[227], const double colWeights_data[4313],
    const unsigned char img_data[2211840], const double ipColIndices_data[4313],
    const int img_size[3], const int partialResize_size[3],
    const double kwidthCol, const int plast,
    unsigned char partialResize_data[490320])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = 3UL * (227UL * (static_cast<unsigned long>(plast - 1) + 1UL)) - 1UL;
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    double sumVal;
    unsigned long tmpIndex;
    int colIdx;
    int colIndices;
    int rowIdx;
    int rowIndices;
    unsigned char u;
    rowIndices = static_cast<int>(idx % 3UL);
    tmpIndex = (idx - static_cast<unsigned long>(rowIndices)) / 3UL;
    colIdx = static_cast<int>(tmpIndex % 227UL);
    tmpIndex = (tmpIndex - static_cast<unsigned long>(colIdx)) / 227UL;
    rowIdx = static_cast<int>(tmpIndex);
    sumVal = 0.0;
    colIndices = static_cast<int>(ceil(kwidthCol));
    for (int l{0}; l < colIndices; l++) {
      sumVal +=
          static_cast<double>(
              img_data[(rowIdx + img_size[0] *
                                     (static_cast<int>(
                                          ipColIndices_data[colIdx + 227 * l]) -
                                      1)) +
                       img_size[0] * img_size[1] * rowIndices]) *
          (colWeights_data[colIdx + 227 * l] / colWeightsTotal[colIdx]);
    }
    sumVal = round(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }
    partialResize_data[(rowIdx + partialResize_size[0] * colIdx) +
                       partialResize_size[0] * 227 * rowIndices] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeightsTotal[227]
//                const double rowWeights_data[2951]
//                const unsigned char partialResize_data[490320]
//                const int partialResize_size[3]
//                const double ipRowIndices_data[2951]
//                const double kwidthRow
//                unsigned char out[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void myAlexNetGPU_kernel18(
    const double rowWeightsTotal[227], const double rowWeights_data[2951],
    const unsigned char partialResize_data[490320],
    const int partialResize_size[3], const double ipRowIndices_data[2951],
    const double kwidthRow, unsigned char out[154587])
{
  unsigned long threadId;
  int colIdx;
  int rowIdx;
  int rowIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  rowIndices = static_cast<int>(threadId % 3UL);
  threadId = (threadId - static_cast<unsigned long>(rowIndices)) / 3UL;
  rowIdx = static_cast<int>(threadId % 227UL);
  threadId = (threadId - static_cast<unsigned long>(rowIdx)) / 227UL;
  colIdx = static_cast<int>(threadId);
  if ((static_cast<int>((static_cast<int>(colIdx < 227)) &&
                        (static_cast<int>(rowIdx < 227)))) &&
      (static_cast<int>(rowIndices < 3))) {
    double sumVal;
    int colIndices;
    unsigned char u;
    sumVal = 0.0;
    colIndices = static_cast<int>(ceil(kwidthRow));
    for (int l{0}; l < colIndices; l++) {
      sumVal +=
          static_cast<double>(
              partialResize_data[((static_cast<int>(
                                       ipRowIndices_data[rowIdx + 227 * l]) +
                                   partialResize_size[0] * colIdx) +
                                  partialResize_size[0] * 227 * rowIndices) -
                                 1]) *
          (rowWeights_data[rowIdx + 227 * l] / rowWeightsTotal[rowIdx]);
    }
    sumVal = round(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }
    out[(rowIdx + 227 * colIdx) + 51529 * rowIndices] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rowWeightsTotal[227]
//                const double rowWeights_data[2951]
//                const unsigned char img_data[2211840]
//                const int img_size[3]
//                const double ipRowIndices_data[2951]
//                const int partialResize_size[3]
//                const double kwidthRow
//                const int plast
//                unsigned char partialResize_data[697344]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel19(
    const double rowWeightsTotal[227], const double rowWeights_data[2951],
    const unsigned char img_data[2211840], const int img_size[3],
    const double ipRowIndices_data[2951], const int partialResize_size[3],
    const double kwidthRow, const int plast,
    unsigned char partialResize_data[697344])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = 3UL * (227UL * (static_cast<unsigned long>(plast - 1) + 1UL)) - 1UL;
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    double sumVal;
    unsigned long tmpIndex;
    int colIdx;
    int colIndices;
    int rowIdx;
    int rowIndices;
    unsigned char u;
    rowIndices = static_cast<int>(idx % 3UL);
    tmpIndex = (idx - static_cast<unsigned long>(rowIndices)) / 3UL;
    rowIdx = static_cast<int>(tmpIndex % 227UL);
    tmpIndex = (tmpIndex - static_cast<unsigned long>(rowIdx)) / 227UL;
    colIdx = static_cast<int>(tmpIndex);
    sumVal = 0.0;
    colIndices = static_cast<int>(ceil(kwidthRow));
    for (int l{0}; l < colIndices; l++) {
      sumVal +=
          static_cast<double>(
              img_data[((static_cast<int>(ipRowIndices_data[rowIdx + 227 * l]) +
                         img_size[0] * colIdx) +
                        img_size[0] * img_size[1] * rowIndices) -
                       1]) *
          (rowWeights_data[rowIdx + 227 * l] / rowWeightsTotal[rowIdx]);
    }
    sumVal = round(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }
    partialResize_data[(rowIdx + 227 * colIdx) +
                       227 * partialResize_size[1] * rowIndices] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char I_data[]
//                const int I_size
//                unsigned char img_data[2211840]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel2(
    const unsigned char I_data[], const int I_size,
    unsigned char img_data[2211840])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(I_size);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    img_data[oldIdx] = I_data[oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double colWeightsTotal[227]
//                const double colWeights_data[4313]
//                const unsigned char partialResize_data[697344]
//                const int partialResize_size[3]
//                const double ipColIndices_data[4313]
//                const double kwidthCol
//                unsigned char out[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void myAlexNetGPU_kernel20(
    const double colWeightsTotal[227], const double colWeights_data[4313],
    const unsigned char partialResize_data[697344],
    const int partialResize_size[3], const double ipColIndices_data[4313],
    const double kwidthCol, unsigned char out[154587])
{
  unsigned long threadId;
  int colIdx;
  int rowIdx;
  int rowIndices;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  rowIndices = static_cast<int>(threadId % 3UL);
  threadId = (threadId - static_cast<unsigned long>(rowIndices)) / 3UL;
  rowIdx = static_cast<int>(threadId % 227UL);
  threadId = (threadId - static_cast<unsigned long>(rowIdx)) / 227UL;
  colIdx = static_cast<int>(threadId);
  if ((static_cast<int>((static_cast<int>(colIdx < 227)) &&
                        (static_cast<int>(rowIdx < 227)))) &&
      (static_cast<int>(rowIndices < 3))) {
    double sumVal;
    int colIndices;
    unsigned char u;
    sumVal = 0.0;
    colIndices = static_cast<int>(ceil(kwidthCol));
    for (int l{0}; l < colIndices; l++) {
      sumVal +=
          static_cast<double>(
              partialResize_data
                  [(rowIdx + 227 * (static_cast<int>(
                                        ipColIndices_data[colIdx + 227 * l]) -
                                    1)) +
                   227 * partialResize_size[1] * rowIndices]) *
          (colWeights_data[colIdx + 227 * l] / colWeightsTotal[colIdx]);
    }
    sumVal = round(sumVal);
    if (sumVal < 256.0) {
      if (sumVal >= 0.0) {
        u = static_cast<unsigned char>(sumVal);
      } else {
        u = static_cast<unsigned char>(0U);
      }
    } else if (sumVal >= 256.0) {
      u = MAX_uint8_T;
    } else {
      u = static_cast<unsigned char>(0U);
    }
    out[(rowIdx + 227 * colIdx) + 51529 * rowIndices] = u;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char img_data[2211840]
//                const int img_size[3]
//                const int b_img_size[3]
//                const int plast
//                const int in_rows
//                unsigned char b_img_data[2211840]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel3(
    const unsigned char img_data[2211840], const int img_size[3],
    const int b_img_size[3], const int plast, const int in_rows,
    unsigned char b_img_data[2211840])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = (static_cast<unsigned long>(plast) + 1UL) *
                ((static_cast<unsigned long>(in_rows) + 1UL) * 3UL) -
            1UL;
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    unsigned long tmpIndex;
    int colIndices;
    int oldIdx;
    int rowIndices;
    colIndices =
        static_cast<int>(idx % (static_cast<unsigned long>(plast) + 1UL));
    tmpIndex = (idx - static_cast<unsigned long>(colIndices)) /
               (static_cast<unsigned long>(plast) + 1UL);
    rowIndices = static_cast<int>(tmpIndex %
                                  (static_cast<unsigned long>(in_rows) + 1UL));
    tmpIndex = (tmpIndex - static_cast<unsigned long>(rowIndices)) /
               (static_cast<unsigned long>(in_rows) + 1UL);
    oldIdx = static_cast<int>(tmpIndex);
    b_img_data[(colIndices + b_img_size[0] * rowIndices) +
               b_img_size[0] * b_img_size[1] * oldIdx] =
        img_data[(colIndices + img_size[0] * rowIndices) +
                 img_size[0] * img_size[1] * (2 - oldIdx)];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned char img_data[2211840]
//                const int img_size
//                unsigned char b_img_data[2211840]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel4(
    const unsigned char img_data[2211840], const int img_size,
    unsigned char b_img_data[2211840])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(img_size);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    b_img_data[oldIdx] = img_data[oldIdx];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int in_rows
//                const int plast
//                double aux1_data[1440]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel5(
    const int in_rows, const int plast, double aux1_data[1440])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int colIndices;
    colIndices = static_cast<int>(idx);
    if (colIndices + 1 <= in_rows) {
      aux1_data[colIndices] = static_cast<double>(colIndices) + 1.0;
    } else {
      aux1_data[colIndices] =
          (static_cast<double>(plast - colIndices) - 1.0) + 1.0;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int in_cols
//                const int plast
//                double aux2_data[2048]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel6(
    const int in_cols, const int plast, double aux2_data[2048])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int colIndices;
    colIndices = static_cast<int>(idx);
    if (colIndices + 1 <= in_cols) {
      aux2_data[colIndices] = static_cast<double>(colIndices) + 1.0;
    } else {
      aux2_data[colIndices] =
          (static_cast<double>(plast - colIndices) - 1.0) + 1.0;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int plast
//                double ipRowIndices_data[2951]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel7(
    const int plast, double ipRowIndices_data[2951])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(227 * plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    ipRowIndices_data[oldIdx] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int plast
//                double rowWeights_data[2951]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void myAlexNetGPU_kernel8(
    const int plast, double rowWeights_data[2951])
{
  unsigned long loopEnd;
  unsigned long threadId;
  unsigned long threadStride;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  threadStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long>(227 * plast - 1);
  for (unsigned long idx{threadId}; idx <= loopEnd; idx += threadStride) {
    int oldIdx;
    oldIdx = static_cast<int>(idx);
    rowWeights_data[oldIdx] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double aux1_data[1440]
//                const int in_rows
//                const double scale_idx_0
//                const double kwidthRow
//                double ipRowIndices_data[2951]
//                double rowWeights_data[2951]
// Return Type  : void
//
static __global__ __launch_bounds__(256, 1) void myAlexNetGPU_kernel9(
    const double aux1_data[1440], const int in_rows, const double scale_idx_0,
    const double kwidthRow, double ipRowIndices_data[2951],
    double rowWeights_data[2951])
{
  unsigned long threadId;
  int rowIdx;
  threadId = static_cast<unsigned long>(mwGetGlobalThreadIndexInXDimension());
  rowIdx = static_cast<int>(threadId);
  if (rowIdx < 227) {
    int colIndices;
    colIndices = static_cast<int>(ceil(kwidthRow));
    for (int k{0}; k < colIndices; k++) {
      double absx;
      double absx2;
      double sumVal;
      int b_y;
      int l;
      int rowIndices;
      sumVal = (static_cast<double>(rowIdx) + 1.0) / scale_idx_0 +
               0.5 * (1.0 - 1.0 / scale_idx_0);
      rowIndices = static_cast<int>(floor(sumVal - kwidthRow / 2.0));
      sumVal -= static_cast<double>(rowIndices + k) + 1.0;
      if (scale_idx_0 < 1.0) {
        sumVal *= scale_idx_0;
      }
      absx = fabs(sumVal);
      absx2 = absx * absx;
      sumVal = pow(absx, 3.0);
      sumVal = ((1.5 * sumVal - 2.5 * absx2) + 1.0) *
                   static_cast<double>(absx <= 1.0) +
               (((-0.5 * sumVal + 2.5 * absx2) - 4.0 * absx) + 2.0) *
                   static_cast<double>((static_cast<int>(1.0 < absx)) &&
                                       (static_cast<int>(absx <= 2.0)));
      if (scale_idx_0 < 1.0) {
        rowWeights_data[rowIdx + 227 * k] = scale_idx_0 * sumVal;
      } else {
        rowWeights_data[rowIdx + 227 * k] = sumVal;
      }
      b_y = in_rows << 1;
      rowIndices = (rowIndices + k) + 1;
      l = rowIndices - 1;
      if (b_y == 0) {
        if (rowIndices - 1 == 0) {
          l = 0;
        }
      } else if (rowIndices - 1 == 0) {
        l = 0;
      } else {
        l = static_cast<int>(fmod(static_cast<double>(rowIndices) - 1.0,
                                  static_cast<double>(b_y)));
        if ((static_cast<int>(l != 0)) &&
            (static_cast<int>(rowIndices - 1 < 0))) {
          l += b_y;
        }
      }
      ipRowIndices_data[rowIdx + 227 * k] = aux1_data[l];
    }
  }
}

//
// MYALEXNETGPU Accepts a 227x227x3 image to the deep neural network AlexNet
//  and returns the class index of the maximum confidence classification.
//
//  A list of all the classifications can be found in the file
//  classificationList.txt
//
//  Copyright 2018 The MathWorks, Inc.
//
// Arguments    : const unsigned char I_data[]
//                const int I_size[3]
// Return Type  : double
//
double myAlexNetGPU(const unsigned char I_data[], const int I_size[3])
{
  dim3 block;
  dim3 grid;
  double(*gpu_colWeights_data)[4313];
  double(*gpu_ipColIndices_data)[4313];
  double(*gpu_ipRowIndices_data)[2951];
  double(*gpu_rowWeights_data)[2951];
  double(*gpu_aux2_data)[2048];
  double(*gpu_aux1_data)[1440];
  double(*gpu_colWeightsTotal)[227];
  double(*gpu_rowWeightsTotal)[227];
  double classIdx;
  double kwidthCol;
  double kwidthRow;
  double scale_idx_0;
  double scale_idx_1;
  float output[1000];
  float ex;
  int b_img_size[3];
  int img_size[3];
  int partialResize_size[3];
  int(*b_gpu_img_size)[3];
  int(*gpu_I_size)[3];
  int(*gpu_img_size)[3];
  int(*gpu_partialResize_size)[3];
  int colWeights_size[2];
  int rowWeights_size[2];
  int in_cols;
  int plast;
  unsigned char(*b_gpu_img_data)[2211840];
  unsigned char(*gpu_img_data)[2211840];
  unsigned char(*b_gpu_partialResize_data)[697344];
  unsigned char(*gpu_partialResize_data)[490320];
  unsigned char out[154587];
  unsigned char(*gpu_out)[154587];
  unsigned char *gpu_I_data;
  bool b;
  bool validLaunchParams;
  if (!isInitialized_myAlexNetGPU) {
    myAlexNetGPU_initialize();
  }
  cudaMalloc(&gpu_out, 154587UL);
  cudaMalloc(&b_gpu_partialResize_data, 697344UL);
  cudaMalloc(&gpu_partialResize_data, 490320UL);
  cudaMalloc(&gpu_partialResize_size, 12UL);
  cudaMalloc(&gpu_colWeightsTotal, 1816UL);
  cudaMalloc(&gpu_rowWeightsTotal, 1816UL);
  cudaMalloc(&gpu_colWeights_data, 34504UL);
  cudaMalloc(&gpu_ipColIndices_data, 34504UL);
  cudaMalloc(&gpu_rowWeights_data, 23608UL);
  cudaMalloc(&gpu_ipRowIndices_data, 23608UL);
  cudaMalloc(&gpu_aux2_data, 16384UL);
  cudaMalloc(&gpu_aux1_data, 11520UL);
  cudaMalloc(&b_gpu_img_data, 2211840UL);
  cudaMalloc(&b_gpu_img_size, 12UL);
  cudaMalloc(&gpu_img_size, 12UL);
  cudaMalloc(&gpu_I_size, 12UL);
  cudaMalloc(&gpu_img_data, 2211840UL);
  cudaMalloc(&gpu_I_data, 2211840U * sizeof(unsigned char));
  //  Since the function "alexnet" is not supported for generation we load it
  //  from a MAT-file using coder.loadDeepLearningNetwork
  if (!net_not_empty) {
    coder::DeepLearningNetwork_setup(&net);
    net.matlabCodegenIsDeleted = false;
    net_not_empty = true;
  }
  //  Convert image data format OpenCV BGR to MATLAB compatible RGB image format
  //  Copyright 2020 The MathWorks, Inc.
  b = true;
  if ((I_size[1] != 0) && (I_size[2] != 0)) {
    bool exitg1;
    plast = 0;
    in_cols = 1;
    exitg1 = false;
    while ((!exitg1) && (in_cols < 4)) {
      if (I_size[3 - in_cols] != 1) {
        if (plast > 4 - in_cols) {
          b = false;
          exitg1 = true;
        } else {
          plast = 4 - in_cols;
          in_cols++;
        }
      } else {
        in_cols++;
      }
    }
  }
  if (b) {
    img_size[0] = I_size[2];
    img_size[1] = I_size[1];
    b = true;
    plast = I_size[1] * I_size[2] * 3 - 1;
    validLaunchParams = mwGetLaunchParameters1D(static_cast<double>(plast + 1L),
                                                &grid, &block, 1024U, 65535U);
    if (validLaunchParams) {
      cudaMemcpy(gpu_I_data, I_data,
                 3 * (I_size[1] * I_size[2]) * sizeof(unsigned char),
                 cudaMemcpyHostToDevice);
      myAlexNetGPU_kernel2<<<grid, block>>>(gpu_I_data, plast, *gpu_img_data);
    }
  } else {
    img_size[0] = I_size[2];
    img_size[1] = I_size[1];
    b = true;
    in_cols = I_size[2];
    plast = I_size[1] - 1;
    validLaunchParams = mwGetLaunchParameters1D(
        static_cast<double>(3L *
                            (((I_size[1] - 1) + 1L) * ((I_size[2] - 1) + 1L))),
        &grid, &block, 1024U, 65535U);
    if (validLaunchParams) {
      cudaMemcpy(gpu_I_data, I_data,
                 3 * (I_size[1] * I_size[2]) * sizeof(unsigned char),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(*gpu_I_size, I_size, 12UL, cudaMemcpyHostToDevice);
      cudaMemcpy(*gpu_img_size, img_size, 12UL, cudaMemcpyHostToDevice);
      b = false;
      myAlexNetGPU_kernel1<<<grid, block>>>(gpu_I_data, *gpu_I_size,
                                            *gpu_img_size, plast, in_cols,
                                            *gpu_img_data);
    }
  }
  plast = img_size[0] - 1;
  in_cols = img_size[1] - 1;
  b_img_size[0] = img_size[0];
  b_img_size[1] = img_size[1];
  validLaunchParams = mwGetLaunchParameters1D(
      static_cast<double>(((img_size[0] - 1) + 1L) *
                          (((img_size[1] - 1) + 1L) * 3L)),
      &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    if (b) {
      cudaMemcpy(*gpu_img_size, img_size, 12UL, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(*b_gpu_img_size, b_img_size, 12UL, cudaMemcpyHostToDevice);
    myAlexNetGPU_kernel3<<<grid, block>>>(*gpu_img_data, *gpu_img_size,
                                          *b_gpu_img_size, plast, in_cols,
                                          *b_gpu_img_data);
  }
  img_size[0] = b_img_size[0];
  img_size[1] = b_img_size[1];
  plast = b_img_size[0] * b_img_size[1] * 3 - 1;
  validLaunchParams = mwGetLaunchParameters1D(static_cast<double>(plast + 1L),
                                              &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel4<<<grid, block>>>(*b_gpu_img_data, plast,
                                          *gpu_img_data);
  }
  // sz = size(img);
  // sizeWH = sz([2 1]);
  //  Resize
  scale_idx_0 = 227.0 / static_cast<double>(b_img_size[0]);
  scale_idx_1 = 227.0 / static_cast<double>(b_img_size[1]);
  plast = b_img_size[0] << 1;
  validLaunchParams = mwGetLaunchParameters1D(
      static_cast<double>((plast - 1) + 1L), &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel5<<<grid, block>>>(b_img_size[0], plast, *gpu_aux1_data);
  }
  plast = b_img_size[1] << 1;
  validLaunchParams = mwGetLaunchParameters1D(
      static_cast<double>((plast - 1) + 1L), &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel6<<<grid, block>>>(b_img_size[1], plast, *gpu_aux2_data);
  }
  if (scale_idx_0 < 1.0) {
    kwidthRow = 4.0 / scale_idx_0;
  } else {
    kwidthRow = 4.0;
  }
  plast = static_cast<int>(std::ceil(kwidthRow));
  validLaunchParams =
      mwGetLaunchParameters1D(static_cast<double>((227 * plast - 1) + 1L),
                              &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel7<<<grid, block>>>(plast, *gpu_ipRowIndices_data);
  }
  plast = static_cast<int>(std::ceil(kwidthRow));
  rowWeights_size[1] = plast;
  validLaunchParams =
      mwGetLaunchParameters1D(static_cast<double>((227 * plast - 1) + 1L),
                              &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel8<<<grid, block>>>(plast, *gpu_rowWeights_data);
  }
  myAlexNetGPU_kernel9<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
      *gpu_aux1_data, b_img_size[0], scale_idx_0, kwidthRow,
      *gpu_ipRowIndices_data, *gpu_rowWeights_data);
  if (scale_idx_1 < 1.0) {
    kwidthCol = 4.0 / scale_idx_1;
  } else {
    kwidthCol = 4.0;
  }
  plast = static_cast<int>(std::ceil(kwidthCol));
  validLaunchParams =
      mwGetLaunchParameters1D(static_cast<double>((227 * plast - 1) + 1L),
                              &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel10<<<grid, block>>>(plast, *gpu_ipColIndices_data);
  }
  plast = static_cast<int>(std::ceil(kwidthCol));
  colWeights_size[1] = plast;
  validLaunchParams =
      mwGetLaunchParameters1D(static_cast<double>((227 * plast - 1) + 1L),
                              &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    myAlexNetGPU_kernel11<<<grid, block>>>(plast, *gpu_colWeights_data);
  }
  myAlexNetGPU_kernel12<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
      *gpu_aux2_data, b_img_size[1], scale_idx_1, kwidthCol,
      *gpu_ipColIndices_data, *gpu_colWeights_data);
  plast = rowWeights_size[1];
  myAlexNetGPU_kernel13<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
      *gpu_rowWeights_data, *gpu_rowWeightsTotal);
  for (in_cols = 0; in_cols <= plast - 2; in_cols++) {
    myAlexNetGPU_kernel14<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
        *gpu_rowWeights_data, (in_cols + 1) * 227, *gpu_rowWeightsTotal);
  }
  plast = colWeights_size[1];
  myAlexNetGPU_kernel15<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
      *gpu_colWeights_data, *gpu_colWeightsTotal);
  for (in_cols = 0; in_cols <= plast - 2; in_cols++) {
    myAlexNetGPU_kernel16<<<dim3(1U, 1U, 1U), dim3(256U, 1U, 1U)>>>(
        *gpu_colWeights_data, (in_cols + 1) * 227, *gpu_colWeightsTotal);
  }
  if (!(scale_idx_0 > scale_idx_1)) {
    partialResize_size[1] = b_img_size[1];
    validLaunchParams = mwGetLaunchParameters1D(
        static_cast<double>(3L * (227L * ((b_img_size[1] - 1) + 1L))), &grid,
        &block, 1024U, 65535U);
    if (validLaunchParams) {
      cudaMemcpy(*gpu_img_size, img_size, 12UL, cudaMemcpyHostToDevice);
      cudaMemcpy(*gpu_partialResize_size, partialResize_size, 12UL,
                 cudaMemcpyHostToDevice);
      myAlexNetGPU_kernel19<<<grid, block>>>(
          *gpu_rowWeightsTotal, *gpu_rowWeights_data, *gpu_img_data,
          *gpu_img_size, *gpu_ipRowIndices_data, *gpu_partialResize_size,
          kwidthRow, b_img_size[1], *b_gpu_partialResize_data);
    } else {
      cudaMemcpy(*gpu_partialResize_size, partialResize_size, 12UL,
                 cudaMemcpyHostToDevice);
    }
    myAlexNetGPU_kernel20<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>(
        *gpu_colWeightsTotal, *gpu_colWeights_data, *b_gpu_partialResize_data,
        *gpu_partialResize_size, *gpu_ipColIndices_data, kwidthCol, *gpu_out);
  } else {
    partialResize_size[0] = b_img_size[0];
    validLaunchParams = mwGetLaunchParameters1D(
        static_cast<double>(3L * (227L * ((b_img_size[0] - 1) + 1L))), &grid,
        &block, 1024U, 65535U);
    if (validLaunchParams) {
      cudaMemcpy(*gpu_img_size, img_size, 12UL, cudaMemcpyHostToDevice);
      cudaMemcpy(*gpu_partialResize_size, partialResize_size, 12UL,
                 cudaMemcpyHostToDevice);
      myAlexNetGPU_kernel17<<<grid, block>>>(
          *gpu_colWeightsTotal, *gpu_colWeights_data, *gpu_img_data,
          *gpu_ipColIndices_data, *gpu_img_size, *gpu_partialResize_size,
          kwidthCol, b_img_size[0], *gpu_partialResize_data);
    } else {
      cudaMemcpy(*gpu_partialResize_size, partialResize_size, 12UL,
                 cudaMemcpyHostToDevice);
    }
    myAlexNetGPU_kernel18<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>(
        *gpu_rowWeightsTotal, *gpu_rowWeights_data, *gpu_partialResize_data,
        *gpu_partialResize_size, *gpu_ipRowIndices_data, kwidthRow, *gpu_out);
  }
  //  Predict with AlexNet
  cudaMemcpy(out, *gpu_out, 154587UL, cudaMemcpyDeviceToHost);
  coder::DeepLearningNetwork_predict(&net, out, output);
  //  Determine the class index with the highest probability
  plast = 1;
  ex = output[0];
  for (in_cols = 0; in_cols < 999; in_cols++) {
    float f;
    f = output[in_cols + 1];
    if (std::isnan(f)) {
      b = false;
    } else if (std::isnan(ex)) {
      b = true;
    } else {
      b = (ex < f);
    }
    if (b) {
      ex = f;
      plast = in_cols + 2;
    }
  }
  classIdx = plast;
  cudaFree(gpu_I_data);
  cudaFree(*gpu_img_data);
  cudaFree(*gpu_I_size);
  cudaFree(*gpu_img_size);
  cudaFree(*b_gpu_img_size);
  cudaFree(*b_gpu_img_data);
  cudaFree(*gpu_aux1_data);
  cudaFree(*gpu_aux2_data);
  cudaFree(*gpu_ipRowIndices_data);
  cudaFree(*gpu_rowWeights_data);
  cudaFree(*gpu_ipColIndices_data);
  cudaFree(*gpu_colWeights_data);
  cudaFree(*gpu_rowWeightsTotal);
  cudaFree(*gpu_colWeightsTotal);
  cudaFree(*gpu_partialResize_size);
  cudaFree(*gpu_partialResize_data);
  cudaFree(*b_gpu_partialResize_data);
  cudaFree(*gpu_out);
  return classIdx;
}

//
// MYALEXNETGPU Accepts a 227x227x3 image to the deep neural network AlexNet
//  and returns the class index of the maximum confidence classification.
//
//  A list of all the classifications can be found in the file
//  classificationList.txt
//
//  Copyright 2018 The MathWorks, Inc.
//
// Arguments    : void
// Return Type  : void
//
void myAlexNetGPU_free()
{
  if (!net.matlabCodegenIsDeleted) {
    net.matlabCodegenIsDeleted = true;
    coder::DeepLearningNetwork_delete(&net);
  }
}

//
// MYALEXNETGPU Accepts a 227x227x3 image to the deep neural network AlexNet
//  and returns the class index of the maximum confidence classification.
//
//  A list of all the classifications can be found in the file
//  classificationList.txt
//
//  Copyright 2018 The MathWorks, Inc.
//
// Arguments    : void
// Return Type  : void
//
void myAlexNetGPU_init()
{
  net_not_empty = false;
  net.matlabCodegenIsDeleted = true;
}

//
// File trailer for myAlexNetGPU.cu
//
// [EOF]
//
