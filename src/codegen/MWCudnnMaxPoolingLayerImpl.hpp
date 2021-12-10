/* Copyright 2020-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_MAX_POOLING_LAYER_IMPL
#define MW_CUDNN_MAX_POOLING_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"

class MWCNNLayer;

namespace MWCudnnTarget
{
class MWTargetNetworkImpl;

//MaxPooling2DLayer
class MWMaxPoolingLayerImpl: public MWCNNLayerImpl
{
public:
    //Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
    MWMaxPoolingLayerImpl(MWCNNLayer *, MWTargetNetworkImpl*, int, int, int, int, int, int, int, int, bool, int);
    ~MWMaxPoolingLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();
    void allocate();
    void deallocate();
    float* getIndexData();
    
public:
    int DCdZnqpcBnvXVgEsLBnz;
    int DqxLTLaJwwgQqmrtCDuu;
    int CufLFODQDXTAPyRqYodN;
    int DSsxcjIrUgZCKZovyNQf;

    int CGbFsczkgkhjcHoCKzBx;
    int CDJtexcMbXMWAmnNZsNf;
    int CZNYmBcNFSZWvaCklqeM;
    int ClEhcJFlvGCgiavziIag;

    int FLuSVNoPhAFKtLUchSvv;
    int FeVcBgtQmTLtmnNcJGMY;

    bool BLjrjqvCcCommiXWQLjs;
    
private:
    float* URgvgDXnZskIYGdtimcU;
    float* OwortPcLToImGdYFtbSF;
    int etjQLJVQCaeAXRWYtqOl;
    cudnnPoolingDescriptor_t lHtftnmGBvlSSoGOXVui;
    float puSFZkRJmyuFPfQRswDK;
    bool vFNECEAeLZsYsUxvlgqL;
    cudnnTensorDescriptor_t DRzwhbNPpftRRIXXfHzd;
};

} // namespace MWCudnnTarget

#endif
