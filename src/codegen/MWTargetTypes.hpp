#ifndef MW_TARGET_TYPES_HPP
#define MW_TARGET_TYPES_HPP

namespace MWTargetType {

enum TARGETTYPE {
    CUDNN_TARGET = 0,
    TENSORRT_TARGET,
    MKLDNN_TARGET,
    ARMNEON_TARGET,
    ARMMALI_TARGET
};

} // namespace MWTargetType

#endif
