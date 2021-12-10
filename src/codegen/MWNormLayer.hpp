/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_NORM_LAYER
#define MW_NORM_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// CrossChannelNormalizationLayer
class MWNormLayer : public MWCNNLayer {
  public:
    MWNormLayer() {
    }
    ~MWNormLayer() {
    }

    void
    createNormLayer(MWTargetNetworkImplBase*, MWTensorBase*, unsigned, double, double, double, const char*, int);
    void propagateSize();

  private:
    unsigned windowChannelSize;
    double alpha;
    double beta;
    double k;
};

#endif
