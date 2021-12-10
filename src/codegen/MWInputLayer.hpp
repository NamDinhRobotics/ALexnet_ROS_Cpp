/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_INPUT_LAYER
#define MW_INPUT_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

// ImageInputLayer
class MWInputLayer : public MWCNNLayer {
  public:
    MWInputLayer() {
    }
    ~MWInputLayer() {
    }

    void createInputLayer(MWTargetNetworkImplBase* ntwk_impl,
                          MWTensorBase* m_in,
                          const char*,
                          int);
    void propagateSize();
};
#endif
