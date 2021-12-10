/* Copyright 2020-2021 The MathWorks, Inc. */
#ifndef MW_OUTPUT_LAYER
#define MW_OUTPUT_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// ClassificationOutputLayer
class MWOutputLayer : public MWCNNLayer {
  public:
    MWOutputLayer() {
    }
    ~MWOutputLayer() {
    }

    void createOutputLayer(MWTargetNetworkImplBase*, MWTensorBase*, const char*, int);
    void predict();
    void propagateSize();
};
#endif
