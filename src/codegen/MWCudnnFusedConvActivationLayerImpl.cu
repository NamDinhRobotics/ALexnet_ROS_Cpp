#include "MWCudnnFusedConvActivationLayerImpl.hpp"
#include "MWCudnnCNNLayerImpl.hpp"
#include "MWCNNLayer.hpp"
#include "MWTensorBase.hpp"
#include "MWTensor.hpp"
#include "MWCudnnTargetNetworkImpl.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
 namespace MWCudnnTarget { 
MWFusedConvActivationLayerImpl::MWFusedConvActivationLayerImpl(MWCNNLayer* 
layer, MWTargetNetworkImpl* ntwk_impl, int filt_H, int filt_W, int numGrps, int 
numChnls, int numFilts, int FOcStuqCptsGIZXskVpC, int 
FpguQZSermqZCMRiUfML, int CTCbzQMDaLxINPbODdng, int 
CLOUhPjbgggWoXHTtmjC, int CpMjJjtGOeWOzwxpAAQP, int 
CqtPRJvHlGJFssiPzsOm, int AjhVZuQXURJimwbnYqDF, int 
AwZQzUhuWVLGrWgLHRuM, int edQOkUJIZbwzEeIcCLzG, const char* 
xHViLEwTujGGrPZZgmbF, const char* JxwPQNPACGfmGpNncpCY, double , 
MWActivationFunctionType::ACTIVATION_FCN_ENUM GZGFVDrXwFLJleoTDywO) : 
MWCNNLayerImpl(layer, ntwk_impl) , vIWQzNvYZSuxmOTVDFhU(NULL) , IwKnaBoXVubIRYcxEJLH(NULL) , 
TfsmDFpPPOscKZifVzSQ(NULL) , WprSrhAStKGxyXeoxETy(NULL) , 
FshVHIJMRAhtQirYPlZd(NULL) , HgeIbZCtKXtKFOEtSlPZ(NULL) , 
AzTsxYcYjIEJsGQbeYHm(filt_H) , BHuHNDGoRwGRouCxeMbw (filt_W) , 
BkwhtPQUCQKchmmimoXs (numGrps) , BUOdotSvmFyUWQKMUdra (numChnls) , 
BdqURaHPmdnfzvtUvocl (numFilts) , FLuSVNoPhAFKtLUchSvv(FOcStuqCptsGIZXskVpC) 
, FeVcBgtQmTLtmnNcJGMY(FpguQZSermqZCMRiUfML) , 
CGbFsczkgkhjcHoCKzBx(CTCbzQMDaLxINPbODdng) , 
CDJtexcMbXMWAmnNZsNf(CLOUhPjbgggWoXHTtmjC) , 
CZNYmBcNFSZWvaCklqeM(CpMjJjtGOeWOzwxpAAQP) , 
ClEhcJFlvGCgiavziIag(CqtPRJvHlGJFssiPzsOm) , 
AdmgfUbRAfzFeYHxSnQr(AjhVZuQXURJimwbnYqDF) , 
AuqaQHxmPQSyYRemQvyX(AwZQzUhuWVLGrWgLHRuM) , 
fSKMHAqIghbYYgyIpNDw(edQOkUJIZbwzEeIcCLzG) , 
IAlDgIFcchbwRGBSfVfA((CGbFsczkgkhjcHoCKzBx != CDJtexcMbXMWAmnNZsNf) 
|| (CZNYmBcNFSZWvaCklqeM != ClEhcJFlvGCgiavziIag)) { 
assert(GZGFVDrXwFLJleoTDywO == MWActivationFunctionType::ACTIVATION_FCN_ENUM::RELU);
#if (CUDNN_MAJOR < 6)
 throw std::runtime_error("Fused ConvReLU Layer only supported for cuDNN 6 or greater");
#else
 cQBKlCKXxecGPJrXBXdk = ntwk_impl; 
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&NMMfJylfQjiIUAKhXCJb)); 
CUDNN_CALL(cudnnCreateFilterDescriptor(&PtkeOkuClHzhOfpmBevf)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&JsZenQeBPMhwsyEhVHiD));  
CUDNN_CALL(cudnnCreateActivationDescriptor(&muwRQxtWMMXAPxSuMYBw)); 
MWTensorBase* ipTensor_conv = getLayer()->getInputTensor(0); int 
NNhshzQGJHLSGjDiVerE = CGbFsczkgkhjcHoCKzBx; int 
NXruhrCCiguRjAgSNDuz = CZNYmBcNFSZWvaCklqeM; if 
(IAlDgIFcchbwRGBSfVfA) { NNhshzQGJHLSGjDiVerE = 0; 
NXruhrCCiguRjAgSNDuz = 0; TfsmDFpPPOscKZifVzSQ = new MWTensor<float>(-1, 
-1, -1, -1, -1, NULL, getLayer(), ipTensor_conv->getDataFormat(), 0); if 
(!TfsmDFpPPOscKZifVzSQ) { MWCNNLayerImpl::throwAllocationError(__LINE__ , 
__FILE__); } CUDNN_CALL(cudnnCreateTensorDescriptor(&XYbzSmRQGatVJtGmDZSo)); } 
else { TfsmDFpPPOscKZifVzSQ = ipTensor_conv; } assert(TfsmDFpPPOscKZifVzSQ != 
NULL); bERCRkGjpaKXMNComoYl = CGbFsczkgkhjcHoCKzBx; bOrQjJTNlssnrexxbHdi = 
CZNYmBcNFSZWvaCklqeM; 
CUDNN_CALL(cudnnSetConvolution2dDescriptor(NMMfJylfQjiIUAKhXCJb, 
NNhshzQGJHLSGjDiVerE, NXruhrCCiguRjAgSNDuz, FLuSVNoPhAFKtLUchSvv, 
FeVcBgtQmTLtmnNcJGMY, AdmgfUbRAfzFeYHxSnQr, AuqaQHxmPQSyYRemQvyX, 
CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#if (FP16_ENABLED == 1 && ( CUDNN_MAJOR > 7 || (CUDNN_MAJOR == 7 && CUDNN_MINOR >= 2) ))
 CUDNN_CALL(cudnnSetConvolutionMathType(NMMfJylfQjiIUAKhXCJb, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif
 if (BkwhtPQUCQKchmmimoXs > 1){ 
CUDNN_CALL(cudnnSetConvolutionGroupCount(NMMfJylfQjiIUAKhXCJb, 
BkwhtPQUCQKchmmimoXs)); } 
CUDNN_CALL(cudnnSetActivationDescriptor(muwRQxtWMMXAPxSuMYBw, 
CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0)); int 
eWYFXrUazhqiEIscccda = BUOdotSvmFyUWQKMUdra*BkwhtPQUCQKchmmimoXs; int 
eqmVWbEcwBRGnVNDUtrG = BdqURaHPmdnfzvtUvocl*BkwhtPQUCQKchmmimoXs; 
CUDNN_CALL(cudnnSetFilter4dDescriptor(PtkeOkuClHzhOfpmBevf, CUDNN_DATA_FLOAT, 
CUDNN_TENSOR_NCHW, eqmVWbEcwBRGnVNDUtrG, 
eWYFXrUazhqiEIscccda/BkwhtPQUCQKchmmimoXs, AzTsxYcYjIEJsGQbeYHm, 
BHuHNDGoRwGRouCxeMbw)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(JsZenQeBPMhwsyEhVHiD, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, 1, eqmVWbEcwBRGnVNDUtrG, 1, 1)); int weightSize = 
BUOdotSvmFyUWQKMUdra*eqmVWbEcwBRGnVNDUtrG*AzTsxYcYjIEJsGQbeYHm*BHuHNDGoRwGRouCxeMbw; 
CUDA_CALL(cudaMalloc((void**)&vIWQzNvYZSuxmOTVDFhU, sizeof(float)*weightSize)); 
CUDA_CALL(cudaMalloc((void**)&IwKnaBoXVubIRYcxEJLH, 
sizeof(float)*eqmVWbEcwBRGnVNDUtrG)); 
loadWeights(xHViLEwTujGGrPZZgmbF); loadBias(JxwPQNPACGfmGpNncpCY); createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
#endif
 } MWFusedConvActivationLayerImpl::~MWFusedConvActivationLayerImpl() { } void 
MWFusedConvActivationLayerImpl::propagateSize() {
#if (CUDNN_MAJOR >= 6)
 MWTensorBase* ipTensor_conv = getLayer()->getInputTensor(0); int inputH; int 
inputW; if (IAlDgIFcchbwRGBSfVfA) { inputH = 
ipTensor_conv->getHeight() + CGbFsczkgkhjcHoCKzBx + CDJtexcMbXMWAmnNZsNf; 
inputW = ipTensor_conv->getWidth() + CZNYmBcNFSZWvaCklqeM + 
ClEhcJFlvGCgiavziIag; } else { inputH = ipTensor_conv->getHeight(); inputW = 
ipTensor_conv->getWidth(); } TfsmDFpPPOscKZifVzSQ->setHeight(inputH); 
TfsmDFpPPOscKZifVzSQ->setWidth(inputW); 
TfsmDFpPPOscKZifVzSQ->setChannels(ipTensor_conv->getChannels()); 
TfsmDFpPPOscKZifVzSQ->setBatchSize(ipTensor_conv->getBatchSize()); 
TfsmDFpPPOscKZifVzSQ->setSequenceLength(ipTensor_conv->getSequenceLength()); 
assert(TfsmDFpPPOscKZifVzSQ->getSequenceLength() == 1); if 
(IAlDgIFcchbwRGBSfVfA) { 
CUDNN_CALL(cudnnSetTensor4dDescriptor(XYbzSmRQGatVJtGmDZSo, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, TfsmDFpPPOscKZifVzSQ->getBatchSize(), 
TfsmDFpPPOscKZifVzSQ->getChannels(), TfsmDFpPPOscKZifVzSQ->getHeight(), 
TfsmDFpPPOscKZifVzSQ->getWidth())); } else { XYbzSmRQGatVJtGmDZSo = 
MWCNNLayerImpl::getCuDNNDescriptor(TfsmDFpPPOscKZifVzSQ); } 
assert(BUOdotSvmFyUWQKMUdra == 
TfsmDFpPPOscKZifVzSQ->getChannels()/BkwhtPQUCQKchmmimoXs); MWTensorBase* opTensor 
= getLayer()->getOutputTensor(0); cudnnTensorDescriptor_t* desc = 
getDescriptor(opTensor->getSourcePortIndex()); assert(desc); 
setDescriptor<float>(*desc, static_cast<MWTensor<float>*>(opTensor));
#if (CUDNN_MAJOR < 7)
 { 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, PtkeOkuClHzhOfpmBevf, NMMfJylfQjiIUAKhXCJb, *desc, 
CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &NDjzAZSYJuWymuKDNZYB)); }
#else
 { const int maxAlgoCount(3); int returnedAlgoCount(-1); 
cudnnConvolutionFwdAlgoPerf_t perf_results[maxAlgoCount]; 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, PtkeOkuClHzhOfpmBevf, NMMfJylfQjiIUAKhXCJb, *desc, 
maxAlgoCount, &returnedAlgoCount, perf_results)); NDjzAZSYJuWymuKDNZYB = 
perf_results[0].algo; }
#endif
 if (CUDNN_VERSION < 7402) fixConvAlgo(); size_t sxuOMwKXOKfuExclRaSe = 0; 
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, PtkeOkuClHzhOfpmBevf, NMMfJylfQjiIUAKhXCJb, *desc, 
NDjzAZSYJuWymuKDNZYB, &sxuOMwKXOKfuExclRaSe)); if( sxuOMwKXOKfuExclRaSe > 
*cQBKlCKXxecGPJrXBXdk->getProposedWorkSpaceSize() ) { 
cQBKlCKXxecGPJrXBXdk->setProposedWorkSpaceSize(sxuOMwKXOKfuExclRaSe); }
#endif
 } void MWFusedConvActivationLayerImpl::allocate() { MWTensorBase* 
ipTensor_conv = getLayer()->getInputTensor(0); if 
(IAlDgIFcchbwRGBSfVfA) { float* newInput; int inputH = 
ipTensor_conv->getHeight() + CGbFsczkgkhjcHoCKzBx + CDJtexcMbXMWAmnNZsNf; int 
inputW = ipTensor_conv->getWidth() + CZNYmBcNFSZWvaCklqeM + 
ClEhcJFlvGCgiavziIag; int paddedSize = ipTensor_conv->getBatchSize() * 
ipTensor_conv->getChannels() * inputH * inputW; 
CUDA_CALL(cudaMalloc((void**)&newInput, sizeof(float)*paddedSize)); 
CUDA_CALL(cudaMemset(newInput, 0, sizeof(float)*paddedSize)); 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->setData(newInput); } 
WprSrhAStKGxyXeoxETy = 
static_cast<MWTensor<float>*>(getLayer()->getOutputTensor(0))->getData(); 
setalpha2Ptr(getZeroPtr()); int numInputs = getLayer()->getNumInputs(); if 
(numInputs == 2) { setalpha2Ptr(getOnePtr()); WprSrhAStKGxyXeoxETy = 
static_cast<MWTensor<float>*>(getLayer()->getInputTensor(1))->getData(); } if 
(static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData() == 
WprSrhAStKGxyXeoxETy){ int xInputTensorSize = 
getLayer()->getInputTensor(0)->getNumElements(); 
CUDA_CALL(cudaMalloc((void**)&FshVHIJMRAhtQirYPlZd, sizeof(float) * 
xInputTensorSize)); } } void MWFusedConvActivationLayerImpl::deallocate() { if 
(TfsmDFpPPOscKZifVzSQ != getLayer()->getInputTensor(0)) { 
assert(IAlDgIFcchbwRGBSfVfA); 
CUDA_FREE_CALL(static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData()); 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->setData((float*)NULL); } if 
(FshVHIJMRAhtQirYPlZd){ CUDA_FREE_CALL(FshVHIJMRAhtQirYPlZd); 
FshVHIJMRAhtQirYPlZd = NULL;  }  } void 
MWFusedConvActivationLayerImpl::predict() { MWCNNLayer* fusedConvReluLayer = 
getLayer(); MWTensorBase* ipTensorBase = fusedConvReluLayer->getInputTensor(); 
MWTensorBase* opTensorBase = fusedConvReluLayer->getOutputTensor(); 
MWTensor<float>* ipTensor = static_cast<MWTensor<float>*>(ipTensorBase); 
MWTensor<float>* opTensor = static_cast<MWTensor<float>*>(opTensorBase); if 
(TfsmDFpPPOscKZifVzSQ != fusedConvReluLayer->getInputTensor()) { 
CUDA_CALL(cudaMemset(static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData(), 
0, sizeof(float)*TfsmDFpPPOscKZifVzSQ->getNumElements())); 
MWCNNLayerImpl::padInput(ipTensor->getData(), ipTensor->getHeight(), 
ipTensor->getWidth(), ipTensor->getChannels(), TfsmDFpPPOscKZifVzSQ->getHeight(), 
TfsmDFpPPOscKZifVzSQ->getWidth(), bERCRkGjpaKXMNComoYl, bOrQjJTNlssnrexxbHdi, 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData(), 
ipTensor->getNumElements()); } cudnnTensorDescriptor_t* desc = 
getDescriptor(opTensor->getSourcePortIndex()); assert(desc);
#if (CUDNN_MAJOR >= 6)
 assert(opTensor->getData() != 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData() || 
(getLayer()->getNumInputs() == 2)); float* pzUAoBDvaKAtdsmkQuct; if 
(static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData() == 
WprSrhAStKGxyXeoxETy){ CUDA_CALL(cudaMemcpy(FshVHIJMRAhtQirYPlZd, 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData(), sizeof(float) * 
opTensorBase->getNumElements(), cudaMemcpyDeviceToDevice)); 
pzUAoBDvaKAtdsmkQuct = FshVHIJMRAhtQirYPlZd; }else{ 
pzUAoBDvaKAtdsmkQuct = 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData(); } 
CUDNN_CALL(cudnnConvolutionBiasActivationForward(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
getOnePtr(), XYbzSmRQGatVJtGmDZSo, pzUAoBDvaKAtdsmkQuct, 
PtkeOkuClHzhOfpmBevf, vIWQzNvYZSuxmOTVDFhU, NMMfJylfQjiIUAKhXCJb, NDjzAZSYJuWymuKDNZYB, 
cQBKlCKXxecGPJrXBXdk->getWorkSpace(), 
*cQBKlCKXxecGPJrXBXdk->getAllocatedWorkSpaceSize(), getalpha2Ptr(),  *desc,  
WprSrhAStKGxyXeoxETy,  JsZenQeBPMhwsyEhVHiD, IwKnaBoXVubIRYcxEJLH, muwRQxtWMMXAPxSuMYBw, 
*desc, opTensor->getData()));
#endif
 } void MWFusedConvActivationLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyConvolutionDescriptor(NMMfJylfQjiIUAKhXCJb)); 
CUDNN_CALL(cudnnDestroyFilterDescriptor(PtkeOkuClHzhOfpmBevf)); 
CUDNN_CALL(cudnnDestroyActivationDescriptor(muwRQxtWMMXAPxSuMYBw)); if 
(vIWQzNvYZSuxmOTVDFhU) { CUDA_FREE_CALL(vIWQzNvYZSuxmOTVDFhU); vIWQzNvYZSuxmOTVDFhU = NULL; } 
CUDNN_CALL(cudnnDestroyTensorDescriptor(JsZenQeBPMhwsyEhVHiD)); if 
(IwKnaBoXVubIRYcxEJLH) { CUDA_FREE_CALL(IwKnaBoXVubIRYcxEJLH); IwKnaBoXVubIRYcxEJLH = NULL; } if 
(TfsmDFpPPOscKZifVzSQ != getLayer()->getInputTensor(0)) { 
assert(IAlDgIFcchbwRGBSfVfA); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(XYbzSmRQGatVJtGmDZSo)); } } void 
MWFusedConvActivationLayerImpl::loadWeights(const char* PmFfARVzoHVAYkfpuvqK) { 
FILE* QMgBqCuvjnbWHWiVPEwn = MWCNNLayer::openBinaryFile(PmFfARVzoHVAYkfpuvqK); 
assert(QMgBqCuvjnbWHWiVPEwn); int cRtIUoZRPICuQEOZOSzT = 
BUOdotSvmFyUWQKMUdra*BkwhtPQUCQKchmmimoXs*BdqURaHPmdnfzvtUvocl*AzTsxYcYjIEJsGQbeYHm*BHuHNDGoRwGRouCxeMbw; 
 float* KZWeXiYFmdpQdsgidKeG = MALLOC_CALL(sizeof(float)*cRtIUoZRPICuQEOZOSzT); 
MWCNNLayer::call_fread(KZWeXiYFmdpQdsgidKeG, sizeof(float), cRtIUoZRPICuQEOZOSzT, 
QMgBqCuvjnbWHWiVPEwn, PmFfARVzoHVAYkfpuvqK); CUDA_CALL(cudaMemcpy(vIWQzNvYZSuxmOTVDFhU, 
KZWeXiYFmdpQdsgidKeG, sizeof(float)*cRtIUoZRPICuQEOZOSzT, cudaMemcpyHostToDevice));
#if 0
 printf("%s loaded. Size = %d. %f\n", PmFfARVzoHVAYkfpuvqK, cRtIUoZRPICuQEOZOSzT, KZWeXiYFmdpQdsgidKeG[0]);
#endif
 free(KZWeXiYFmdpQdsgidKeG); fclose(QMgBqCuvjnbWHWiVPEwn); return; } void 
MWFusedConvActivationLayerImpl::loadBias(const char* PmFfARVzoHVAYkfpuvqK) { 
FILE* QMgBqCuvjnbWHWiVPEwn = MWCNNLayer::openBinaryFile(PmFfARVzoHVAYkfpuvqK); 
assert(QMgBqCuvjnbWHWiVPEwn); int cRtIUoZRPICuQEOZOSzT = 
BkwhtPQUCQKchmmimoXs*BdqURaHPmdnfzvtUvocl;  float* KZWeXiYFmdpQdsgidKeG = 
MALLOC_CALL(sizeof(float)*cRtIUoZRPICuQEOZOSzT); 
MWCNNLayer::call_fread(KZWeXiYFmdpQdsgidKeG, sizeof(float), cRtIUoZRPICuQEOZOSzT, 
QMgBqCuvjnbWHWiVPEwn, PmFfARVzoHVAYkfpuvqK); CUDA_CALL(cudaMemcpy(IwKnaBoXVubIRYcxEJLH, 
KZWeXiYFmdpQdsgidKeG, sizeof(float)*cRtIUoZRPICuQEOZOSzT, cudaMemcpyHostToDevice)); 
free(KZWeXiYFmdpQdsgidKeG); fclose(QMgBqCuvjnbWHWiVPEwn); return; } void 
MWFusedConvActivationLayerImpl::setLearnables(std::vector<float*> learnables) { 
assert(learnables.size() == 2); int cRtIUoZRPICuQEOZOSzT = BUOdotSvmFyUWQKMUdra * 
BdqURaHPmdnfzvtUvocl * BkwhtPQUCQKchmmimoXs * AzTsxYcYjIEJsGQbeYHm * 
BHuHNDGoRwGRouCxeMbw;  float* wqggPBXZvtlxnxwngvAq = learnables[0]; 
CUDA_CALL(cudaMemcpy(vIWQzNvYZSuxmOTVDFhU, wqggPBXZvtlxnxwngvAq, sizeof(float) * 
cRtIUoZRPICuQEOZOSzT, cudaMemcpyHostToDevice)); cRtIUoZRPICuQEOZOSzT = 
BdqURaHPmdnfzvtUvocl * BkwhtPQUCQKchmmimoXs;  float* JgLfgHrHMEMmMYTettJF = 
learnables[1]; CUDA_CALL(cudaMemcpy(IwKnaBoXVubIRYcxEJLH, JgLfgHrHMEMmMYTettJF, 
sizeof(float) * cRtIUoZRPICuQEOZOSzT, cudaMemcpyHostToDevice)); } void 
MWFusedConvActivationLayerImpl::postSetup() { if 
(cQBKlCKXxecGPJrXBXdk->getAutoTune()) { getConvAlgoTuned(); } else { 
getConvAlgoWorkSpaceLimit(); } } void 
MWFusedConvActivationLayerImpl::getConvAlgoTuned() { MWTensorBase* opTensorBase 
= getLayer()->getOutputTensor(0); MWTensor<float>* opTensor = 
static_cast<MWTensor<float>*>(opTensorBase); cudnnConvolutionFwdAlgoPerf_t 
perf_results[3]; cudnnTensorDescriptor_t* desc = 
getDescriptor(getLayer()->getOutputTensor()->getSourcePortIndex()); 
assert(desc); int returnedAlgoCount; 
CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, 
static_cast<MWTensor<float>*>(TfsmDFpPPOscKZifVzSQ)->getData(), 
PtkeOkuClHzhOfpmBevf, vIWQzNvYZSuxmOTVDFhU, NMMfJylfQjiIUAKhXCJb, *desc, 
opTensor->getData(), 3, &returnedAlgoCount, &perf_results[0], 
cQBKlCKXxecGPJrXBXdk->getWorkSpace(), 
*cQBKlCKXxecGPJrXBXdk->getAllocatedWorkSpaceSize())); NDjzAZSYJuWymuKDNZYB = 
perf_results[0].algo; if (CUDNN_VERSION < 7402) fixConvAlgo(); } void 
MWFusedConvActivationLayerImpl::getConvAlgoWorkSpaceLimit() { 
cudnnTensorDescriptor_t* desc = 
getDescriptor(getLayer()->getOutputTensor()->getSourcePortIndex()); assert(desc);
#if (CUDNN_MAJOR < 8)
 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, PtkeOkuClHzhOfpmBevf, NMMfJylfQjiIUAKhXCJb, *desc, 
CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 
*cQBKlCKXxecGPJrXBXdk->getAllocatedWorkSpaceSize(), &NDjzAZSYJuWymuKDNZYB));
#else
 int maxAlgoCount(-1); 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
&maxAlgoCount)); int returnedAlgoCount(-1); 
std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(maxAlgoCount);  
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
XYbzSmRQGatVJtGmDZSo, PtkeOkuClHzhOfpmBevf, NMMfJylfQjiIUAKhXCJb, *desc, 
maxAlgoCount, &returnedAlgoCount, &perf_results[0])); 
cudnnConvolutionFwdAlgoPerf_t nextFastest; bool algoFound(false); for (int i = 
0; i < returnedAlgoCount; ++i) { nextFastest = perf_results[i]; if 
(nextFastest.memory <= *cQBKlCKXxecGPJrXBXdk->getAllocatedWorkSpaceSize()) { 
NDjzAZSYJuWymuKDNZYB = nextFastest.algo; algoFound = true; break; } } assert(algoFound);
#endif
 if (CUDNN_VERSION < 7402) fixConvAlgo(); } void 
MWFusedConvActivationLayerImpl::fixConvAlgo() { int inputH = 
TfsmDFpPPOscKZifVzSQ->getHeight(); int inputW = TfsmDFpPPOscKZifVzSQ->getWidth(); 
if (NDjzAZSYJuWymuKDNZYB == CUDNN_CONVOLUTION_FWD_ALGO_FFT && (inputH > 64 || 
inputW > 64)) { NDjzAZSYJuWymuKDNZYB = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; 
} } } 