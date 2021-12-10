#include "MWCudnnTargetNetworkImpl.hpp"
#include "MWTargetNetworkImplBase.hpp"
#include "MWTargetTypes.hpp"
#include "MWCudnnCNNLayerImpl.hpp"
#include "MWCudnnLayerImplFactory.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
 namespace MWCudnnTarget { MWTargetNetworkImpl::MWTargetNetworkImpl() : 
MWTargetNetworkImplBase(MWTargetType::CUDNN_TARGET, new 
MWCudnnLayerImplFactory) , xcusoQxPPodcHwVviCWI(0) , 
NZjOkZPwLzQsdEVkwMcX(0) , NbunkIVaMPVYgAQHXXYd(0) , MW_autoTune(true) 
, leWFtIPrKkXLixGWBGJW(0) , GsZlHFuhbvjLtRMDjXnW(0) {} 
void MWTargetNetworkImpl::allocate(int BufSize, int numBufsToAlloc) { numBufs = 
numBufsToAlloc; memBuffer.reserve(numBufs); for(int i = 0; i < numBufs; i++) { 
float *memPtr = 0; CUDA_CALL(cudaMalloc((void**)&memPtr, 
sizeof(float)*BufSize)); memBuffer.push_back(memPtr); }  } void 
MWTargetNetworkImpl::allocatePermuteBuffers(int bufSize, int numBufsToAlloc) { 
for (int i = 0; i < numBufsToAlloc; i++) { float* memPtr = 0; 
CUDA_CALL(cudaMalloc((void**)&memPtr, sizeof(float) * bufSize)); 
kqftrrQBBOgGsrDSkIUk.push_back(memPtr); } } void 
MWTargetNetworkImpl::preSetup() {  NZjOkZPwLzQsdEVkwMcX = new 
cublasHandle_t; if(!NZjOkZPwLzQsdEVkwMcX) { 
MWCNNLayerImpl::throwAllocationError(__LINE__ , __FILE__); } 
CUBLAS_CALL(cublasCreate(NZjOkZPwLzQsdEVkwMcX)); NbunkIVaMPVYgAQHXXYd 
= new cudnnHandle_t; if(!NbunkIVaMPVYgAQHXXYd) { 
MWCNNLayerImpl::throwAllocationError(__LINE__ , __FILE__); } 
CUDNN_CALL(cudnnCreate(NbunkIVaMPVYgAQHXXYd));  } void 
MWTargetNetworkImpl::postSetup(MWCNNLayer* layers[],int numLayers) { if 
(*getProposedWorkSpaceSize() > *getAllocatedWorkSpaceSize()) { if 
(xcusoQxPPodcHwVviCWI) { destroyWorkSpace(xcusoQxPPodcHwVviCWI); } 
createWorkSpace(xcusoQxPPodcHwVviCWI); while ((!xcusoQxPPodcHwVviCWI) && 
(*getProposedWorkSpaceSize() > 0)) { 
setProposedWorkSpaceSize(MWTargetNetworkImpl::getNextProposedWorkSpaceSize(*getProposedWorkSpaceSize())); 
createWorkSpace(xcusoQxPPodcHwVviCWI); } } for (int i = 0; i < numLayers; i++) 
{ layers[i]->postSetup();  }  } size_t 
MWTargetNetworkImpl::getNextProposedWorkSpaceSize(size_t failedWorkSpaceSize) { 
assert(failedWorkSpaceSize > 0); return failedWorkSpaceSize/2; } void 
MWTargetNetworkImpl::createWorkSpace(float*& xkUNToJIgvoLoUQuzKRF) { 
cudaError_t rMMjgjGRAiLVlTlRSByU = cudaMalloc((void**)&xkUNToJIgvoLoUQuzKRF, 
*getProposedWorkSpaceSize()); if (rMMjgjGRAiLVlTlRSByU != cudaSuccess) { 
xkUNToJIgvoLoUQuzKRF = NULL; setAllocatedWorkSpaceSize(0);  
rMMjgjGRAiLVlTlRSByU = cudaGetLastError();  } else { 
setAllocatedWorkSpaceSize(*getProposedWorkSpaceSize()); } } void 
MWTargetNetworkImpl::destroyWorkSpace(float*& xkUNToJIgvoLoUQuzKRF) { 
CUDA_FREE_CALL(xkUNToJIgvoLoUQuzKRF); xkUNToJIgvoLoUQuzKRF = NULL; 
setAllocatedWorkSpaceSize(0);  } void 
MWTargetNetworkImpl::setProposedWorkSpaceSize(size_t wss) { 
leWFtIPrKkXLixGWBGJW = wss;  } size_t* 
MWTargetNetworkImpl::getProposedWorkSpaceSize() { return 
&leWFtIPrKkXLixGWBGJW; } void 
MWTargetNetworkImpl::setAllocatedWorkSpaceSize(size_t wss) { 
GsZlHFuhbvjLtRMDjXnW = wss;  } size_t* 
MWTargetNetworkImpl::getAllocatedWorkSpaceSize() { return 
&GsZlHFuhbvjLtRMDjXnW; } float* 
MWTargetNetworkImpl::getWorkSpace() { return xcusoQxPPodcHwVviCWI; } float* 
MWTargetNetworkImpl::getPermuteBuffer(int bufIndex) { return 
kqftrrQBBOgGsrDSkIUk[bufIndex]; } cublasHandle_t* 
MWTargetNetworkImpl::getCublasHandle() { return NZjOkZPwLzQsdEVkwMcX; } 
cudnnHandle_t* MWTargetNetworkImpl::getCudnnHandle() { return 
NbunkIVaMPVYgAQHXXYd; } void MWTargetNetworkImpl::setAutoTune(bool 
autotune) { MW_autoTune = autotune; } bool MWTargetNetworkImpl::getAutoTune() 
const { return MW_autoTune; } void MWTargetNetworkImpl::deallocate() { for(int 
i = 0; i < memBuffer.size(); i++) { float *memPtr = memBuffer[i]; if(memPtr) { 
CUDA_FREE_CALL(memPtr); }  } memBuffer.clear(); for(int i = 0; i < 
kqftrrQBBOgGsrDSkIUk.size(); i++) { float *memPtr = 
kqftrrQBBOgGsrDSkIUk[i]; if(memPtr) { CUDA_FREE_CALL(memPtr); } } 
kqftrrQBBOgGsrDSkIUk.clear(); } void MWTargetNetworkImpl::cleanup() { if 
(xcusoQxPPodcHwVviCWI) { destroyWorkSpace(xcusoQxPPodcHwVviCWI); } if 
(NZjOkZPwLzQsdEVkwMcX) { cudaError_t cudaError = cudaPeekAtLastError(); if 
(cudaError != cudaErrorCudartUnloading) { 
CUBLAS_CALL(cublasDestroy(*NZjOkZPwLzQsdEVkwMcX)); } delete 
NZjOkZPwLzQsdEVkwMcX; } if (NbunkIVaMPVYgAQHXXYd) {
#if (CUDNN_MAJOR < 8) 
 CUDNN_CALL(cudnnDestroy(*NbunkIVaMPVYgAQHXXYd));
#else
 cudaError_t cudaError = cudaPeekAtLastError();  if (cudaError != 
cudaErrorCudartUnloading) { 
CUDNN_CALL(cudnnDestroy(*NbunkIVaMPVYgAQHXXYd)); }
#endif
 delete NbunkIVaMPVYgAQHXXYd; } } float* 
MWTargetNetworkImpl::getBufferPtr(int bufferIndex) { 
assert(static_cast<size_t>(bufferIndex) < memBuffer.size()); return 
memBuffer[bufferIndex]; } } 