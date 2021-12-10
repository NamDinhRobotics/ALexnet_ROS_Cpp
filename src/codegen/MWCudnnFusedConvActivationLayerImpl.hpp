/* Copyright 2018-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_FUSED_CONV_ACTIVATION_LAYER_IMPL
#define MW_CUDNN_FUSED_CONV_ACTIVATION_LAYER_IMPL

#include "MWCudnnCNNLayerImpl.hpp"
#include "MWActivationFunctionType.hpp"

#include <vector>

class MWCNNLayer;

namespace MWCudnnTarget
{
class MWTargetNetworkImpl;

class MWFusedConvActivationLayerImpl: public MWCNNLayerImpl
{

private:
    int AzTsxYcYjIEJsGQbeYHm;           //Filter height for CONV and FC
    int BHuHNDGoRwGRouCxeMbw;            //Filter width for CONV and FC

    int BkwhtPQUCQKchmmimoXs;
    int BUOdotSvmFyUWQKMUdra;
    int BdqURaHPmdnfzvtUvocl;

    int FLuSVNoPhAFKtLUchSvv;
    int FeVcBgtQmTLtmnNcJGMY;
    int CGbFsczkgkhjcHoCKzBx;
    int CDJtexcMbXMWAmnNZsNf;
    int CZNYmBcNFSZWvaCklqeM;
    int ClEhcJFlvGCgiavziIag;
    int AdmgfUbRAfzFeYHxSnQr;
    int AuqaQHxmPQSyYRemQvyX;

    int fSKMHAqIghbYYgyIpNDw;

    float* vIWQzNvYZSuxmOTVDFhU;
    float* IwKnaBoXVubIRYcxEJLH;
    MWTensorBase* TfsmDFpPPOscKZifVzSQ; // for pre-padded input
    int bERCRkGjpaKXMNComoYl;
    int bOrQjJTNlssnrexxbHdi;

    float* WprSrhAStKGxyXeoxETy;

    float* HgeIbZCtKXtKFOEtSlPZ; // Scaling factor for addition

    bool IAlDgIFcchbwRGBSfVfA;

    // Temporary buffer for Xinput, CuDNN 8 upgrade
    float* FshVHIJMRAhtQirYPlZd;    
    
public:
    MWFusedConvActivationLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*,
                                   int, int,
                                   int, int, int,
                                   int, int,
                                   int, int, int, int,
                                   int, int,
                                   int,
                                   const char*, const char*,
                                   double, MWActivationFunctionType::ACTIVATION_FCN_ENUM);
    ~MWFusedConvActivationLayerImpl();

    void predict();
    void cleanup();
    void propagateSize();
    void allocate();
    void deallocate();
    void postSetup();
    void setLearnables(std::vector<float*>);

private:
    void loadWeights(const char*);
    void loadBias(const char*);
    void getConvAlgoTuned();
    void getConvAlgoWorkSpaceLimit();
    void fixConvAlgo(); // g1916490

    void setalpha2Ptr(float* alpha2){ HgeIbZCtKXtKFOEtSlPZ = alpha2;}
    float* getalpha2Ptr() {return HgeIbZCtKXtKFOEtSlPZ;}


private:
    cudnnConvolutionDescriptor_t  NMMfJylfQjiIUAKhXCJb;
    cudnnConvolutionFwdAlgo_t     NDjzAZSYJuWymuKDNZYB;

    cudnnFilterDescriptor_t       PtkeOkuClHzhOfpmBevf;
    cudnnTensorDescriptor_t       JsZenQeBPMhwsyEhVHiD;

    cudnnTensorDescriptor_t      XYbzSmRQGatVJtGmDZSo;

    cudnnActivationDescriptor_t   muwRQxtWMMXAPxSuMYBw;

};
} // namespace MWCudnnTarget
#endif
