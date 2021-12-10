/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_NORM_LAYER_IMPL
#define MW_CUDNN_NORM_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"

class MWCNNLayer;

namespace MWCudnnTarget
{
class MWTargetNetworkImpl;

class MWNormLayerImpl: public MWCNNLayerImpl
{
public:
    MWNormLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, unsigned, double, double, double);
    ~MWNormLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();

private:
    cudnnLRNDescriptor_t          cCXqPFPPcoHzYMDpnUxQ;
};

} // namespace MWCudnnTarget

#endif
