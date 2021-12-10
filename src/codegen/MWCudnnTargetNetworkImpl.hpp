/* Copyright 2017-2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_TARGET_NETWORK_IMPL
#define MW_CUDNN_TARGET_NETWORK_IMPL

#include "MWTargetNetworkImplBase.hpp"

#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>

class MWCNNLayer;

namespace MWCudnnTarget
{

class MWTargetNetworkImpl : public MWTargetNetworkImplBase {

  public:  
    MWTargetNetworkImpl();
    ~MWTargetNetworkImpl() {}
    void allocate(int, int);
    void deallocate();
    void preSetup();
    void postSetup(MWCNNLayer* layers[],int numLayers);
    void cleanup();

    void setProposedWorkSpaceSize(size_t);  // Set the proposed workspace size of the target network impl
    size_t* getProposedWorkSpaceSize();     // Get the proposed workspace size of the target network impl

    void setAllocatedWorkSpaceSize(size_t);  // Set the allocated workspace size of the target network impl 
    size_t* getAllocatedWorkSpaceSize();     // Get the allocated workspace size of the target network impl
    
    float* getWorkSpace();          // Get the workspace buffer in GPU memory    
    cublasHandle_t* getCublasHandle();      // Get the cuBLAS handle to use for GPU computation
    cudnnHandle_t* getCudnnHandle();        // Get the cuDNN handle to use for GPU computation    

    void setAutoTune(bool);
    bool getAutoTune() const;
    float* getBufferPtr(int bufferIndex);

    float* getPermuteBuffer(int index);    // Get the buffer in GPU memory for custom layers' data layout permutation
    void allocatePermuteBuffers(int, int); // allocate buffer for custom layers' data layout permutation

    std::vector<float*> memBuffer;
       
  private:    
    size_t leWFtIPrKkXLixGWBGJW;
    size_t GsZlHFuhbvjLtRMDjXnW;
    float* xcusoQxPPodcHwVviCWI;    
    cublasHandle_t* NZjOkZPwLzQsdEVkwMcX;
    cudnnHandle_t* NbunkIVaMPVYgAQHXXYd;
    bool MW_autoTune;
    long int EGsHUnogBQpOwCZJYeUd;

    std::vector<float *> kqftrrQBBOgGsrDSkIUk;
    
  private:
    void createWorkSpace(float*&);  // Create the workspace needed for this layer
    void destroyWorkSpace(float*&);

    static size_t getNextProposedWorkSpaceSize(size_t failedWorkSpaceSize);

  public:
    static void getStrides(const int* dims, int size, int *stride);
    static void getTransformStrides(const int src[],
                                    const int dest[],
                                    const int dims[],
                                    int size,
                                    int strides[]);
};

} // namespace MWCudnnTarget

#endif
