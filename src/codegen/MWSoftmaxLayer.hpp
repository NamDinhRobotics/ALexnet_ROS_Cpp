/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_SOFTMAX_LAYER
#define MW_SOFTMAX_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// SoftmaxLayer
class MWSoftmaxLayer : public MWCNNLayer {
  public:
    MWSoftmaxLayer() {
    }
    ~MWSoftmaxLayer() {
    }

    void createSoftmaxLayer(MWTargetNetworkImplBase*, MWTensorBase*, const char*, int);
    void propagateSize();
};

#endif
