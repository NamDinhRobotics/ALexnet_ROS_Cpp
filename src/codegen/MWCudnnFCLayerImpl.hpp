/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_FC_LAYER_IMPL
#define MW_CUDNN_FC_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"

#include <vector>

class MWCNNLayer;

namespace MWCudnnTarget
{

class MWTargetNetworkImpl;

// FullyConnectedLayer
class MWFCLayerImpl : public MWCNNLayerImpl {
  public:
    int BuyZFXzwOMxcePIbCLfl;

  private:
    int AzTsxYcYjIEJsGQbeYHm;
    int BHuHNDGoRwGRouCxeMbw;
    int BlRIQPyqJZORKENzSdYf;
    float* vIWQzNvYZSuxmOTVDFhU;
    float* vpXxoeEhdEosLSsYXkNG;
    float* IwKnaBoXVubIRYcxEJLH;

    int xHiBGayUfxIpXKkCTDNU;

  public:
    MWFCLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, int, const char*, const char*);
    ~MWFCLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();
    void postSetup();
    void setLearnables(std::vector<float*>);

  private:
    void loadWeights(const char*);
    void loadBias(const char*);
    void prepareWeights(float*);

  private:
    cudnnTensorDescriptor_t JsZenQeBPMhwsyEhVHiD;
};

} // namespace MWCudnnTarget

#endif
