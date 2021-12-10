/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_RELU_LAYER
#define MW_RELU_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// ReLULayer
class MWReLULayer : public MWCNNLayer {
  public:
    MWReLULayer() {
    }
    ~MWReLULayer() {
    }

    template <typename T1, typename T2>
    void createReLULayer(MWTargetNetworkImplBase*, MWTensorBase*, int, const char*, int, const char*, int);
    void propagateSize();
	
};

#endif
