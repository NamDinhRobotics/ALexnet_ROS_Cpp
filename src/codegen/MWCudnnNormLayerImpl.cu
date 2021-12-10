#include "MWCudnnNormLayerImpl.hpp"
#include "MWCudnnCNNLayerImpl.hpp"
#include "MWCNNLayer.hpp"
#include "MWTensorBase.hpp"
#include "MWTensor.hpp"
#include "MWCudnnTargetNetworkImpl.hpp"
 namespace MWCudnnTarget { MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, unsigned FrpxvsDMwwgbpqHXWxmN,  double 
AFQBkxwYGKLsACiDKwRM,  double AHqhysOOIgbDpWZoPUFT,  double BNrGqqHwfmYKIqbDbnjx) : 
MWCNNLayerImpl(layer, ntwk_impl)  { 
CUDNN_CALL(cudnnCreateLRNDescriptor(&cCXqPFPPcoHzYMDpnUxQ)); 
createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex()); 
CUDNN_CALL(cudnnSetLRNDescriptor(cCXqPFPPcoHzYMDpnUxQ, 
FrpxvsDMwwgbpqHXWxmN, AFQBkxwYGKLsACiDKwRM, AHqhysOOIgbDpWZoPUFT, 
BNrGqqHwfmYKIqbDbnjx)); } MWNormLayerImpl::~MWNormLayerImpl() { } void 
MWNormLayerImpl::propagateSize() { MWTensorBase* opTensor = 
getLayer()->getOutputTensor(0); cudnnTensorDescriptor_t* desc = 
getDescriptor(opTensor->getSourcePortIndex()); assert(desc); 
setDescriptor<float>(*desc, static_cast<MWTensor<float>*>(opTensor));  } void 
MWNormLayerImpl::predict() { MWTensorBase* ipTensorBase = 
getLayer()->getInputTensor();  MWTensorBase* opTensorBase = 
getLayer()->getOutputTensor(); MWTensor<float>* ipTensor = 
static_cast<MWTensor<float>*>(ipTensorBase); MWTensor<float>* opTensor = 
static_cast<MWTensor<float>*>(opTensorBase); cudnnTensorDescriptor_t* desc = 
getDescriptor(opTensor->getSourcePortIndex()); assert(desc); 
cudnnTensorDescriptor_t ipDesc = 
MWCNNLayerImpl::getCuDNNDescriptor(ipTensorBase); 
CUDNN_CALL(cudnnLRNCrossChannelForward(*cQBKlCKXxecGPJrXBXdk->getCudnnHandle(), 
cCXqPFPPcoHzYMDpnUxQ, CUDNN_LRN_CROSS_CHANNEL_DIM1, getOnePtr(), ipDesc, 
ipTensor->getData(), getZeroPtr(), *desc, opTensor->getData())); } void 
MWNormLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyLRNDescriptor(cCXqPFPPcoHzYMDpnUxQ)); } } 