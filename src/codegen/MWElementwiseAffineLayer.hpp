/* Copyright 2017-2021 The MathWorks, Inc. */

#ifndef MW_ELEMENTWISE_AFFINE_LAYER
#define MW_ELEMENTWISE_AFFINE_LAYER

#include "MWCNNLayer.hpp"

class MWTargetNetworkImplBase;
class MWTensorBase;

class MWElementwiseAffineLayer : public MWCNNLayer
{
  private:


  public:
    MWElementwiseAffineLayer();
    ~MWElementwiseAffineLayer();
    void createElementwiseAffineLayer(MWTargetNetworkImplBase*, MWTensorBase*, int, int, int, int, int, int, bool, int, int, const char*, const char*, const char*, int);
    void propagateSize();

  private:
    int scaleH;
    int scaleW;
    int scaleC;
    int offsetH;
    int offsetW;
    int offsetC;
    int isClipped;
    int lowerBound;
    int upperBound;
};
#endif

