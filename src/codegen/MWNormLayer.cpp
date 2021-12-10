/* Copyright 2020-2021 The MathWorks, Inc. */

#include "MWTensorBase.hpp"
#include "MWCNNLayer.hpp"
#include "MWNormLayer.hpp"
#include "MWCNNLayerImplBase.hpp"
#include "MWTargetNetworkImplBase.hpp"
#include "MWLayerImplFactory.hpp"

// Create CrossChannelNormalizationLayer
// Parameters here are the same naming as NNT.
void MWNormLayer::createNormLayer(MWTargetNetworkImplBase* ntwk_impl,
                                  MWTensorBase* m_in,
                                  unsigned m_WindowChannelSize,
                                  double m_Alpha,
                                  double m_Beta,
                                  double m_K,
                                  const char* m_outFormat,
                                  int outbufIdx) {
    windowChannelSize = m_WindowChannelSize;
    alpha = m_Alpha;
    beta = m_Beta;
    k = m_K;

    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL, m_outFormat);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    MWLayerImplFactory* factory = ntwk_impl->getLayerImplFactory();
    m_impl = factory->createNormLayerImpl(this, ntwk_impl, m_WindowChannelSize, m_Alpha, m_Beta, m_K);
}

void MWNormLayer::propagateSize() {
    assert(getInputTensor()->getSequenceLength() == 1);

    resizeOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
}
