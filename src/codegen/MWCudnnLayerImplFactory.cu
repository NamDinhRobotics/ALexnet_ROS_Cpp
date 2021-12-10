#include "MWLayerImplFactory.hpp"
#include "MWCudnnLayerImplFactory.hpp"
#include "MWCNNLayerImplBase.hpp"
#include "MWCudnnCNNLayerImpl.hpp"
#include "MWCudnnTargetNetworkImpl.hpp"
 class MWCNNLayer;
#ifndef CREATE_INPUT_LAYER_IMPL_DEFINITION
#define CREATE_INPUT_LAYER_IMPL_DEFINITION

#include "MWCudnnInputLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createInputLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2) {
return new MWCudnnTarget::MWInputLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2));
}

#endif

#ifndef CREATE_ELEMENTWISEAFFINE_LAYER_IMPL_DEFINITION
#define CREATE_ELEMENTWISEAFFINE_LAYER_IMPL_DEFINITION

#include "MWCudnnElementwiseAffineLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createElementwiseAffineLayerImpl(MWCNNLayer* arg1,
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
const char* arg13) {
return new MWCudnnTarget::MWElementwiseAffineLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2),
arg3,
arg4,
arg5,
arg6,
arg7,
arg8,
arg9,
arg10,
arg11,
arg12,
arg13);
}

#endif

#ifndef CREATE_FUSEDCONVACTIVATION_LAYER_IMPL_DEFINITION
#define CREATE_FUSEDCONVACTIVATION_LAYER_IMPL_DEFINITION

#include "MWCudnnFusedConvActivationLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createFusedConvActivationLayerImpl(MWCNNLayer* arg1,
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
MWActivationFunctionType::ACTIVATION_FCN_ENUM arg20) {
return new MWCudnnTarget::MWFusedConvActivationLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2),
arg3,
arg4,
arg5,
arg6,
arg7,
arg8,
arg9,
arg10,
arg11,
arg12,
arg13,
arg14,
arg15,
arg16,
arg17,
arg18,
arg19,
arg20);
}

#endif

#ifndef CREATE_NORM_LAYER_IMPL_DEFINITION
#define CREATE_NORM_LAYER_IMPL_DEFINITION

#include "MWCudnnNormLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createNormLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
unsigned arg3,
double arg4,
double arg5,
double arg6) {
return new MWCudnnTarget::MWNormLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2),
arg3,
arg4,
arg5,
arg6);
}

#endif

#ifndef CREATE_MAXPOOLING_LAYER_IMPL_DEFINITION
#define CREATE_MAXPOOLING_LAYER_IMPL_DEFINITION

#include "MWCudnnMaxPoolingLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createMaxPoolingLayerImpl(MWCNNLayer* arg1,
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
int arg12) {
return new MWCudnnTarget::MWMaxPoolingLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2),
arg3,
arg4,
arg5,
arg6,
arg7,
arg8,
arg9,
arg10,
arg11,
arg12);
}

#endif

#ifndef CREATE_FC_LAYER_IMPL_DEFINITION
#define CREATE_FC_LAYER_IMPL_DEFINITION

#include "MWCudnnFCLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createFCLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2,
int arg3,
int arg4,
const char* arg5,
const char* arg6) {
return new MWCudnnTarget::MWFCLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2),
arg3,
arg4,
arg5,
arg6);
}

#endif

#ifndef CREATE_RELU_LAYER_IMPL_DEFINITION
#define CREATE_RELU_LAYER_IMPL_DEFINITION

#include "MWCudnnReLULayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createReLULayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2) {
return new MWCudnnTarget::MWReLULayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2));
}

#endif

#ifndef CREATE_SOFTMAX_LAYER_IMPL_DEFINITION
#define CREATE_SOFTMAX_LAYER_IMPL_DEFINITION

#include "MWCudnnSoftmaxLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createSoftmaxLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2) {
return new MWCudnnTarget::MWSoftmaxLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2));
}

#endif

#ifndef CREATE_OUTPUT_LAYER_IMPL_DEFINITION
#define CREATE_OUTPUT_LAYER_IMPL_DEFINITION

#include "MWCudnnOutputLayerImpl.hpp"
MWCNNLayerImplBase* MWCudnnLayerImplFactory::createOutputLayerImpl(MWCNNLayer* arg1,
MWTargetNetworkImplBase* arg2) {
return new MWCudnnTarget::MWOutputLayerImpl(arg1,
static_cast<MWCudnnTarget::MWTargetNetworkImpl*>(arg2));
}

#endif
