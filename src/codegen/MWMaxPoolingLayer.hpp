/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_MAX_POOLING_LAYER
#define MW_MAX_POOLING_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

/**
 * Codegen class for MaxPool layer
 */

// MaxPooling2DLayer
class MWMaxPoolingLayer : public MWCNNLayer {
  public:
    MWMaxPoolingLayer()
        : isGlobalAveragePooling(false) {
    }
    ~MWMaxPoolingLayer() {
    }
    // Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]

    template <typename T1, typename T2>
    void createMaxPoolingLayer(MWTargetNetworkImplBase*,
                               MWTensorBase*,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               bool,
                               int,
                               const char*,
                               int,
                               ...);
    void propagateSize();

    size_t getNumInstrumentedOutputs() {
        return 1;
    }

  private:
    int strideH;
    int strideW;

    int poolH;
    int poolW;

    int paddingH_T;
    int paddingH_B;
    int paddingW_L;
    int paddingW_R;

    bool isGlobalAveragePooling;

    bool hasIndices;
};

#endif
