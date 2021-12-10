/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_SOFTMAX_LAYER_IMPL
#define MW_CUDNN_SOFTMAX_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"

class MWCNNLayer;

namespace MWCudnnTarget
{
class MWTargetNetworkImpl;

//SoftmaxLayer
class MWSoftmaxLayerImpl: public MWCNNLayerImpl
{
public:
    MWSoftmaxLayerImpl(MWCNNLayer* , MWTargetNetworkImpl*);
    ~MWSoftmaxLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();

private:
    cudnnLRNDescriptor_t          cCXqPFPPcoHzYMDpnUxQ;
    cudnnTensorDescriptor_t       rytJDHzuydvYOLNNROYf;
    cudnnTensorDescriptor_t       sFIUeCwGDlfadqOrGZHC;
};

} // namespace MWCudnnTarget

#endif
