//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: DeepLearningNetwork.cu
//
// GPU Coder version                    : 2.2
// CUDA/C/C++ source code generated on  : 10-Dec-2021 14:26:27
//

// Include Files
#include "DeepLearningNetwork.h"
#include "myAlexNetGPU_internal_types.h"
#include "MWCNNLayer.hpp"
#include "MWCudnnTargetNetworkImpl.hpp"
#include "MWElementwiseAffineLayer.hpp"
#include "MWFCLayer.hpp"
#include "MWFusedConvActivationLayer.hpp"
#include "MWInputLayer.hpp"
#include "MWMaxPoolingLayer.hpp"
#include "MWNormLayer.hpp"
#include "MWOutputLayer.hpp"
#include "MWReLULayer.hpp"
#include "MWSoftmaxLayer.hpp"
#include "MWTensor.hpp"
#include "MWTensorBase.hpp"
#include <cstdio>
#include <cstdlib>

// Named Constants
const char *errorString{
    "Abnormal termination due to: %s.\nError in %s (line %d)."};

const char *errStringBase{
    "Error during execution of the generated code. %s at line: %d, file: "
    "%s\nExiting program execution ...\n"};

// Function Declarations
static void checkCleanupCudaError(cudaError_t errCode, const char *file,
                                  unsigned int line);

static void checkRunTimeError(const char *errMsg, const char *file,
                              unsigned int line);

namespace coder {
static void DeepLearningNetwork_callDelete(alexnet0_0 *obj);

}

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::allocate()
{
  targetImpl->allocate(290400, 2);
  for (int idx{0}; idx < 19; idx++) {
    layers[idx]->allocate();
  }
  (static_cast<MWTensor<float> *>(inputTensors[0]))
      ->setData(layers[0]->getLayerOutput(0));
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::cleanup()
{
  deallocate();
  for (int idx{0}; idx < 19; idx++) {
    layers[idx]->cleanup();
  }
  if (targetImpl) {
    targetImpl->cleanup();
  }
  isInitialized = false;
  checkCleanupCudaError(cudaGetLastError(), __FILE__, __LINE__);
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::deallocate()
{
  targetImpl->deallocate();
  for (int idx{0}; idx < 19; idx++) {
    layers[idx]->deallocate();
  }
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::postsetup()
{
  targetImpl->postSetup(layers, numLayers);
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::resetState()
{
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::setSize()
{
  for (int idx{0}; idx < 19; idx++) {
    layers[idx]->propagateSize();
  }
  allocate();
  postsetup();
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::setup()
{
  if (isInitialized) {
    resetState();
  } else {
    targetImpl->preSetup();
    targetImpl->setAutoTune(true);
    (static_cast<MWInputLayer *>(layers[0]))
        ->createInputLayer(targetImpl, inputTensors[0], "SSCB", 0);
    (static_cast<MWElementwiseAffineLayer *>(layers[1]))
        ->createElementwiseAffineLayer(
            targetImpl, layers[0]->getOutputTensor(0), 227, 227, 3, 227, 227, 3,
            false, 1, 1,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_data_scale.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_data_offset.bin", "SSCB",
            0);
    (static_cast<MWFusedConvActivationLayer *>(layers[2]))
        ->createFusedConvActivationLayer(
            targetImpl, 1, layers[1]->getOutputTensor(0), 11, 11, 3, 96, 4, 4,
            0, 0, 0, 0, 1, 1, 1,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv1_w.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv1_b.bin", 0.0,
            MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU, "SSCB", 1);
    (static_cast<MWNormLayer *>(layers[3]))
        ->createNormLayer(targetImpl, layers[2]->getOutputTensor(0), 5, 0.0001,
                          0.75, 1.0, "SSCB", 0);
    (static_cast<MWMaxPoolingLayer *>(layers[4]))
        ->createMaxPoolingLayer<float, float>(
            targetImpl, layers[3]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0, 0,
            0, 0, "FLOAT", 1, "SSCB", 1);
    (static_cast<MWFusedConvActivationLayer *>(layers[5]))
        ->createFusedConvActivationLayer(
            targetImpl, 1, layers[4]->getOutputTensor(0), 5, 5, 48, 128, 1, 1,
            2, 2, 2, 2, 1, 1, 2,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv2_w.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv2_b.bin", 0.0,
            MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU, "SSCB", 0);
    (static_cast<MWNormLayer *>(layers[6]))
        ->createNormLayer(targetImpl, layers[5]->getOutputTensor(0), 5, 0.0001,
                          0.75, 1.0, "SSCB", 1);
    (static_cast<MWMaxPoolingLayer *>(layers[7]))
        ->createMaxPoolingLayer<float, float>(
            targetImpl, layers[6]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0, 0,
            0, 0, "FLOAT", 1, "SSCB", 0);
    (static_cast<MWFusedConvActivationLayer *>(layers[8]))
        ->createFusedConvActivationLayer(
            targetImpl, 1, layers[7]->getOutputTensor(0), 3, 3, 256, 384, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv3_w.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv3_b.bin", 0.0,
            MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU, "SSCB", 1);
    (static_cast<MWFusedConvActivationLayer *>(layers[9]))
        ->createFusedConvActivationLayer(
            targetImpl, 1, layers[8]->getOutputTensor(0), 3, 3, 192, 192, 1, 1,
            1, 1, 1, 1, 1, 1, 2,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv4_w.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv4_b.bin", 0.0,
            MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU, "SSCB", 0);
    (static_cast<MWFusedConvActivationLayer *>(layers[10]))
        ->createFusedConvActivationLayer(
            targetImpl, 1, layers[9]->getOutputTensor(0), 3, 3, 192, 128, 1, 1,
            1, 1, 1, 1, 1, 1, 2,
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv5_w.bin",
            "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_conv5_b.bin", 0.0,
            MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU, "SSCB", 1);
    (static_cast<MWMaxPoolingLayer *>(layers[11]))
        ->createMaxPoolingLayer<float, float>(
            targetImpl, layers[10]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0, 0,
            0, 0, "FLOAT", 1, "SSCB", 0);
    (static_cast<MWFCLayer *>(layers[12]))
        ->createFCLayer(targetImpl, layers[11]->getOutputTensor(0), 9216, 4096,
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc6_w.bin",
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc6_b.bin",
                        "SSCB", 1);
    (static_cast<MWReLULayer *>(layers[13]))
        ->createReLULayer<float, float>(targetImpl,
                                        layers[12]->getOutputTensor(0), 0,
                                        "FLOAT", 1, "SSCB", 1);
    (static_cast<MWFCLayer *>(layers[14]))
        ->createFCLayer(targetImpl, layers[13]->getOutputTensor(0), 4096, 4096,
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc7_w.bin",
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc7_b.bin",
                        "SSCB", 0);
    (static_cast<MWReLULayer *>(layers[15]))
        ->createReLULayer<float, float>(targetImpl,
                                        layers[14]->getOutputTensor(0), 0,
                                        "FLOAT", 1, "SSCB", 0);
    (static_cast<MWFCLayer *>(layers[16]))
        ->createFCLayer(targetImpl, layers[15]->getOutputTensor(0), 4096, 1000,
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc8_w.bin",
                        "./codegen/lib/myAlexNetGPU/cnn_alexnet0_0_fc8_b.bin",
                        "SSCB", 1);
    (static_cast<MWSoftmaxLayer *>(layers[17]))
        ->createSoftmaxLayer(targetImpl, layers[16]->getOutputTensor(0), "SSCB",
                             0);
    (static_cast<MWOutputLayer *>(layers[18]))
        ->createOutputLayer(targetImpl, layers[17]->getOutputTensor(0), "SSCB",
                            0);
    outputTensors[0] = layers[18]->getOutputTensor(0);
    setSize();
  }
  isInitialized = true;
}

//
// Arguments    : cudaError_t errCode
//                const char *file
//                unsigned int line
// Return Type  : void
//
static void checkCleanupCudaError(cudaError_t errCode, const char *file,
                                  unsigned int line)
{
  if ((errCode != cudaSuccess) && (errCode != cudaErrorCudartUnloading)) {
    printf(errorString, cudaGetErrorString(errCode), file, line);
  }
}

//
// Arguments    : const char *errMsg
//                const char *file
//                unsigned int line
// Return Type  : void
//
static void checkRunTimeError(const char *errMsg, const char *file,
                              unsigned int line)
{
  printf(errStringBase, errMsg, line, file);
  exit(EXIT_FAILURE);
}

//
// Arguments    : alexnet0_0 *obj
// Return Type  : void
//
namespace coder {
static void DeepLearningNetwork_callDelete(alexnet0_0 *obj)
{
  if (&obj->isInitialized) {
    obj->cleanup();
  }
}

//
// Arguments    : void
// Return Type  : ::alexnet0_0
//
} // namespace coder
alexnet0_0::alexnet0_0()
{
  numLayers = 19;
  isInitialized = false;
  targetImpl = 0;
  layers[0] = new MWInputLayer;
  layers[0]->setName("data");
  layers[1] = new MWElementwiseAffineLayer;
  layers[1]->setName("data_normalization");
  layers[1]->setInPlaceIndex(0, 0);
  layers[2] = new MWFusedConvActivationLayer;
  layers[2]->setName("conv1_relu1");
  layers[3] = new MWNormLayer;
  layers[3]->setName("norm1");
  layers[4] = new MWMaxPoolingLayer;
  layers[4]->setName("pool1");
  layers[5] = new MWFusedConvActivationLayer;
  layers[5]->setName("conv2_relu2");
  layers[6] = new MWNormLayer;
  layers[6]->setName("norm2");
  layers[7] = new MWMaxPoolingLayer;
  layers[7]->setName("pool2");
  layers[8] = new MWFusedConvActivationLayer;
  layers[8]->setName("conv3_relu3");
  layers[9] = new MWFusedConvActivationLayer;
  layers[9]->setName("conv4_relu4");
  layers[10] = new MWFusedConvActivationLayer;
  layers[10]->setName("conv5_relu5");
  layers[11] = new MWMaxPoolingLayer;
  layers[11]->setName("pool5");
  layers[12] = new MWFCLayer;
  layers[12]->setName("fc6");
  layers[13] = new MWReLULayer;
  layers[13]->setName("relu6");
  layers[13]->setInPlaceIndex(0, 0);
  layers[14] = new MWFCLayer;
  layers[14]->setName("fc7");
  layers[15] = new MWReLULayer;
  layers[15]->setName("relu7");
  layers[15]->setInPlaceIndex(0, 0);
  layers[16] = new MWFCLayer;
  layers[16]->setName("fc8");
  layers[17] = new MWSoftmaxLayer;
  layers[17]->setName("prob");
  layers[18] = new MWOutputLayer;
  layers[18]->setName("output");
  layers[18]->setInPlaceIndex(0, 0);
  targetImpl = new MWCudnnTarget::MWTargetNetworkImpl;
  inputTensors[0] = new MWTensor<float>;
  inputTensors[0]->setHeight(227);
  inputTensors[0]->setWidth(227);
  inputTensors[0]->setChannels(3);
  inputTensors[0]->setBatchSize(1);
  inputTensors[0]->setSequenceLength(1);
}

//
// Arguments    : void
// Return Type  : void
//
alexnet0_0::~alexnet0_0()
{
  try {
    if (isInitialized) {
      cleanup();
    }
    for (int idx{0}; idx < 19; idx++) {
      delete layers[idx];
    }
    if (targetImpl) {
      delete targetImpl;
    }
    delete inputTensors[0];
  } catch (...) {
  }
}

//
// Arguments    : void
// Return Type  : int
//
int alexnet0_0::getBatchSize()
{
  return inputTensors[0]->getBatchSize();
}

//
// Arguments    : int b_index
// Return Type  : float *
//
float *alexnet0_0::getInputDataPointer(int b_index)
{
  return (static_cast<MWTensor<float> *>(inputTensors[b_index]))->getData();
}

//
// Arguments    : void
// Return Type  : float *
//
float *alexnet0_0::getInputDataPointer()
{
  return (static_cast<MWTensor<float> *>(inputTensors[0]))->getData();
}

//
// Arguments    : int layerIndex
//                int portIndex
// Return Type  : float *
//
float *alexnet0_0::getLayerOutput(int layerIndex, int portIndex)
{
  return layers[layerIndex]->getLayerOutput(portIndex);
}

//
// Arguments    : int layerIndex
//                int portIndex
// Return Type  : int
//
int alexnet0_0::getLayerOutputSize(int layerIndex, int portIndex)
{
  return layers[layerIndex]->getOutputTensor(portIndex)->getNumElements() *
         sizeof(float);
}

//
// Arguments    : int b_index
// Return Type  : float *
//
float *alexnet0_0::getOutputDataPointer(int b_index)
{
  return (static_cast<MWTensor<float> *>(outputTensors[b_index]))->getData();
}

//
// Arguments    : void
// Return Type  : float *
//
float *alexnet0_0::getOutputDataPointer()
{
  return (static_cast<MWTensor<float> *>(outputTensors[0]))->getData();
}

//
// Arguments    : void
// Return Type  : void
//
void alexnet0_0::predict()
{
  for (int idx{0}; idx < 19; idx++) {
    layers[idx]->predict();
  }
}

//
// Arguments    : alexnet0_0 *obj
// Return Type  : void
//
namespace coder {
void DeepLearningNetwork_delete(alexnet0_0 *obj)
{
  DeepLearningNetwork_callDelete(obj);
}

//
// Arguments    : alexnet0_0 *obj
// Return Type  : void
//
void DeepLearningNetwork_setup(alexnet0_0 *obj)
{
  try {
    obj->setup();
  } catch (std::runtime_error const &err) {
    obj->cleanup();
    checkRunTimeError(err.what(), __FILE__, __LINE__);
  } catch (...) {
    obj->cleanup();
    checkRunTimeError("", __FILE__, __LINE__);
  }
}

} // namespace coder

//
// File trailer for DeepLearningNetwork.cu
//
// [EOF]
//
