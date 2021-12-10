/* Copyright 2019-2021 The MathWorks, Inc. */
#ifndef MW_CUDNN_ELEMENTWISE_AFFINE_LAYER_IMPL
#define MW_CUDNN_ELEMENTWISE_AFFINE_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"

/**
  *  Codegen class for Scaling Factor layer
**/
class MWCNNLayer;

namespace MWCudnnTarget
{

class MWTargetNetworkImpl;

class MWElementwiseAffineLayerImpl : public MWCNNLayerImpl
{
  public:
    
    MWElementwiseAffineLayerImpl(MWCNNLayer* layer,
                                 MWTargetNetworkImpl* ntwk_impl,
                                 int scale_H,
                                 int scale_W,
                                 int scale_C,
                                 int offset_H,
                                 int offset_W,
                                 int offset_C,
                                 bool isClipped,
                                 int lowerbound,
                                 int upperbound,
                                 const char* scale_file,
                                 const char* offsetfile);
    ~MWElementwiseAffineLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();

  private:
      
    void loadScale(const char*);
    void loadOffset(const char*);
    float* olKGEIcsxmLSoMhRhEtP;
    float* fYaOQTeunPwVjnhhTECh;
    double pFoPPXxxFRbjXXxQWItv;
    double pckLLTEdVPoCZLRwyDnM;
    double osBZbKVTgXwTSsGSbdth;
    double gTcJMwtYuwiqqUmqvKhT;
    double gcGbhKACQPAogUYXHedj;
    double gNROjwaqhxDPvBWUCUcQ;
    bool ZDWLzHUkuZuIUZHfbGDY;
    int bDTIjtxZiSHtjwzgEluE;
    int unSXtdjDjpysqxmbIiPv;
    
    
};

} // namespace MWCudnnTarget

#endif
