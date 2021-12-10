//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: myAlexNetGPU_internal_types.h
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

#ifndef MYALEXNETGPU_INTERNAL_TYPES_H
#define MYALEXNETGPU_INTERNAL_TYPES_H

// Include Files
#include "myAlexNetGPU_types.h"
#include "rtwtypes.h"
#include "MWCNNLayer.hpp"
#include "MWCudnnTargetNetworkImpl.hpp"
#include "MWTensorBase.hpp"

// Type Definitions
class alexnet0_0 {
public:
  alexnet0_0();
  void setSize();
  void resetState();
  void setup();
  void predict();
  void cleanup();
  float *getLayerOutput(int layerIndex, int portIndex);
  int getLayerOutputSize(int layerIndex, int portIndex);
  float *getInputDataPointer(int b_index);
  float *getInputDataPointer();
  float *getOutputDataPointer(int b_index);
  float *getOutputDataPointer();
  int getBatchSize();
  ~alexnet0_0();

private:
  void allocate();
  void postsetup();
  void deallocate();

public:
  bool isInitialized;
  bool matlabCodegenIsDeleted;

private:
  int numLayers;
  MWTensorBase *inputTensors[1];
  MWTensorBase *outputTensors[1];
  MWCNNLayer *layers[19];
  MWCudnnTarget::MWTargetNetworkImpl *targetImpl;
};

#endif
//
// File trailer for myAlexNetGPU_internal_types.h
//
// [EOF]
//
