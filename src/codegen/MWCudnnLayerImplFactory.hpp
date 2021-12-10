/* Copyright 2021 The MathWorks, Inc. */

#ifndef MW_CUDNN_LAYER_IMPL_FACTORY
#define MW_CUDNN_LAYER_IMPL_FACTORY

#include "MWLayerImplFactory.hpp"

class MWCNNLayerImplBase;

class MWCudnnLayerImplFactory : public MWLayerImplFactory {

  public:
    
    MWCudnnLayerImplFactory() {}
    virtual ~MWCudnnLayerImplFactory() {}

    // auto-emit createLayerImpl declarations here

#ifndef CREATE_INPUT_LAYER_IMPL_DECLARATION
#define CREATE_INPUT_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createInputLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2);


#endif

#ifndef CREATE_ELEMENTWISEAFFINE_LAYER_IMPL_DECLARATION
#define CREATE_ELEMENTWISEAFFINE_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createElementwiseAffineLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
int arg3,
int arg4,
int arg5,
int arg6,
int arg7,
int arg8,
bool arg9,
int arg10,
int arg11,
const char* arg12,
const char* arg13);


#endif

#ifndef CREATE_FUSEDCONVACTIVATION_LAYER_IMPL_DECLARATION
#define CREATE_FUSEDCONVACTIVATION_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createFusedConvActivationLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
int arg3,
int arg4,
int arg5,
int arg6,
int arg7,
int arg8,
int arg9,
int arg10,
int arg11,
int arg12,
int arg13,
int arg14,
int arg15,
int arg16,
const char* arg17,
const char* arg18,
double arg19,
MWActivationFunctionType::ACTIVATION_FCN_ENUM arg20);


#endif

#ifndef CREATE_NORM_LAYER_IMPL_DECLARATION
#define CREATE_NORM_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createNormLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
unsigned arg3,
double arg4,
double arg5,
double arg6);


#endif

#ifndef CREATE_MAXPOOLING_LAYER_IMPL_DECLARATION
#define CREATE_MAXPOOLING_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createMaxPoolingLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
int arg3,
int arg4,
int arg5,
int arg6,
int arg7,
int arg8,
int arg9,
int arg10,
bool arg11,
int arg12);


#endif

#ifndef CREATE_FC_LAYER_IMPL_DECLARATION
#define CREATE_FC_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createFCLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
int arg3,
int arg4,
const char* arg5,
const char* arg6);


#endif

#ifndef CREATE_RELU_LAYER_IMPL_DECLARATION
#define CREATE_RELU_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createReLULayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2);


#endif

#ifndef CREATE_SOFTMAX_LAYER_IMPL_DECLARATION
#define CREATE_SOFTMAX_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createSoftmaxLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2);


#endif

#ifndef CREATE_OUTPUT_LAYER_IMPL_DECLARATION
#define CREATE_OUTPUT_LAYER_IMPL_DECLARATION

MWCNNLayerImplBase* createOutputLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2);


#endif
};
#endif