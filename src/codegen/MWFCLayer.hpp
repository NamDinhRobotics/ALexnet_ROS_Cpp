/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_FC_LAYER
#define MW_FC_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// FullyConnectedLayer
class MWFCLayer : public MWCNNLayer {
  public:
    MWFCLayer() {
    }
    ~MWFCLayer() {
    }

    void createFCLayer(MWTargetNetworkImplBase*, MWTensorBase*, int, int, const char*, const char*, const char*, int);
    void propagateSize();
    void setLearnables(float*, float*);

  private:
    int numInputFeatures;
    int numOutputFeatures;
};

#endif
