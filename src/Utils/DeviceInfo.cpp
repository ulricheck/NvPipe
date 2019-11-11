//
// Created by netlabs on 11/9/19.
//

#include "NvPipe/Utils/DeviceInfo.h"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include "NvPipe/Utils/NvCodecUtils.h"

static const struct {
    NVENCSTATUS nverr;
    int         averr;
    const char *desc;
} nvenc_errors[] = {
        { NV_ENC_SUCCESS,                       0, "success"                  },
        { NV_ENC_ERR_NO_ENCODE_DEVICE,         -1, "no encode device"         },
        { NV_ENC_ERR_UNSUPPORTED_DEVICE,       -1, "unsupported device"       },
        { NV_ENC_ERR_INVALID_ENCODERDEVICE,    -1, "invalid encoder device"   },
        { NV_ENC_ERR_INVALID_DEVICE,           -1, "invalid device"           },
        { NV_ENC_ERR_DEVICE_NOT_EXIST,         -1, "device does not exist"    },
        { NV_ENC_ERR_INVALID_PTR,              -1, "invalid ptr"              },
        { NV_ENC_ERR_INVALID_EVENT,            -1, "invalid event"            },
        { NV_ENC_ERR_INVALID_PARAM,            -1, "invalid param"            },
        { NV_ENC_ERR_INVALID_CALL,             -1, "invalid call"             },
        { NV_ENC_ERR_OUT_OF_MEMORY,            -1, "out of memory"            },
        { NV_ENC_ERR_ENCODER_NOT_INITIALIZED,  -1, "encoder not initialized"  },
        { NV_ENC_ERR_UNSUPPORTED_PARAM,        -1, "unsupported param"        },
        { NV_ENC_ERR_LOCK_BUSY,                -1, "lock busy"                },
        { NV_ENC_ERR_NOT_ENOUGH_BUFFER,        -1, "not enough buffer"        },
        { NV_ENC_ERR_INVALID_VERSION,          -1, "invalid version"          },
        { NV_ENC_ERR_MAP_FAILED,               -1, "map failed"               },
        { NV_ENC_ERR_NEED_MORE_INPUT,          -1, "need more input"          },
        { NV_ENC_ERR_ENCODER_BUSY,             -1, "encoder busy"             },
        { NV_ENC_ERR_EVENT_NOT_REGISTERD,      -1, "event not registered"     },
        { NV_ENC_ERR_GENERIC,                  -1, "generic error"            },
        { NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY,  -1, "incompatible client key"  },
        { NV_ENC_ERR_UNIMPLEMENTED,            -1, "unimplemented"            },
        { NV_ENC_ERR_RESOURCE_REGISTER_FAILED, -1, "resource register failed" },
        { NV_ENC_ERR_RESOURCE_NOT_REGISTERED,  -1, "resource not registered"  },
        { NV_ENC_ERR_RESOURCE_NOT_MAPPED,      -1, "resource not mapped"      },
};


#define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
static int nvenc_map_error(NVENCSTATUS err, const char **desc)
{
    int i;
    for (i = 0; i < FF_ARRAY_ELEMS(nvenc_errors); i++) {
        if (nvenc_errors[i].nverr == err) {
            if (desc)
                *desc = nvenc_errors[i].desc;
            return nvenc_errors[i].averr;
        }
    }
    if (desc)
        *desc = "unknown error";
    return -1;
}

static int nvenc_print_error(NVENCSTATUS err,
                             const char *error_string)
{
    const char *desc;
    int ret;
    ret = nvenc_map_error(err, &desc);
    printf("%s: %s (%d)\n", error_string, desc, err);
    return ret;
}

static int check_nv(NVENCSTATUS err, const char *func)
{
    const char *err_string;

    if (err == NV_ENC_SUCCESS) {
        return 0;
    }

    nvenc_map_error(err, &err_string);

    fprintf(stderr, "%s failed", func);
    if (err_string) {
        fprintf(stderr, " -> %s", err_string);
    }
    fprintf(stderr, "\n");

    return -1;
}

#define CHECK_NV(x) { int ret = check_nv((x), #x); if (ret != 0) { return ret; } }

typedef struct {
    NV_ENC_CAPS cap;
    const char *desc;
} cap_t;


static const cap_t nvenc_limits[] = {
        { NV_ENC_CAPS_WIDTH_MAX,                      "Maximum Width" },
        { NV_ENC_CAPS_HEIGHT_MAX,                     "Maximum Hight" },
        { NV_ENC_CAPS_MB_NUM_MAX,                     "Maximum Macroblocks/frame" },
        { NV_ENC_CAPS_MB_PER_SEC_MAX,                 "Maximum Macroblocks/second" },
        { NV_ENC_CAPS_LEVEL_MAX,                      "Max Encoding Level" },
        { NV_ENC_CAPS_LEVEL_MIN,                      "Min Encoding Level" },
        { NV_ENC_CAPS_NUM_MAX_BFRAMES,                "Max No. of B-Frames" },
        { NV_ENC_CAPS_NUM_MAX_LTR_FRAMES,             "Maxmimum LT Reference Frames" },
};

static const cap_t nvenc_caps[] = {
        { NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,    "Supported Rate-Control Modes" },
        { NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,         "Supports Field-Encoding" },
        { NV_ENC_CAPS_SUPPORT_MONOCHROME,             "Supports Monochrome" },
        { NV_ENC_CAPS_SUPPORT_FMO,                    "Supports FMO" },
        { NV_ENC_CAPS_SUPPORT_QPELMV,                 "Supports QPEL Motion Estimation" },
        { NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,           "Supports BDirect Mode" },
        { NV_ENC_CAPS_SUPPORT_CABAC,                  "Supports CABAC" },
        { NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,     "Supports Adaptive Transform" },
        { NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,        "Supports Temporal Layers" },
        { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES,   "Supports Hierarchical P-Frames" },
        { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES,   "Supports Hierarchical B-Frames" },
        { NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,          "Supports Separate Colour Planes" },
        { NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC,           "Supports Temporal SVC" },
        { NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,         "Supports Dynamic Resolution Change" },
        { NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,     "Supports Dynamic Bitrate Change" },
        { NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,      "Supports Dynamic Force Const-QP" },
        { NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,      "Supports Dynamic RC-Mode Change" },
        { NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,      "Supports Sub-Frame Read-back" },
        { NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING,   "Supports Constrained Encoding" },
        { NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,          "Supports Intra Refresh" },
        { NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,    "Supports Custom VBV Buffer Size" },
        { NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,     "Supports Dynamic Slice Mode" },
        { NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION,   "Supports Ref Pic Invalidation" },
        { NV_ENC_CAPS_PREPROC_SUPPORT,                "Supports PreProcessing" },
        { NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,           "Supports Async Encoding" },
        { NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,          "Supports YUV444 Encoding" },
        { NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,        "Supports Lossless Encoding" },
        { NV_ENC_CAPS_SUPPORT_SAO,                    "Supports SAO" },
        { NV_ENC_CAPS_SUPPORT_MEONLY_MODE,            "Supports ME-Only Mode" },
        { NV_ENC_CAPS_SUPPORT_LOOKAHEAD,              "Supports Lookahead Encoding" },
        { NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,            "Supports Temporal AQ" },
        { NV_ENC_CAPS_SUPPORT_10BIT_ENCODE,           "Supports 10-bit Encoding" },
        { NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,    "Supports Weighted Prediction" },
#if 0
/* This isn't really a capability. It's a runtime measurement. */
  { NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY, "Remaining Encoder Capacity" },
#endif
        { NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE,        "Supports B-Frames as References" },
        { NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP,     "Supports Emphasis Level Map" },
#if 0
        { NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,    "Supports Multiple Reference Frames" },
#endif
};

static const struct {
    NV_ENC_BUFFER_FORMAT fmt;
    const char *desc;
} nvenc_formats[] = {
        { NV_ENC_BUFFER_FORMAT_NV12,         "NV12" },
        { NV_ENC_BUFFER_FORMAT_YV12,         "YV12" },
        { NV_ENC_BUFFER_FORMAT_IYUV,         "IYUV" },
        { NV_ENC_BUFFER_FORMAT_YUV444,       "YUV444" },
        { NV_ENC_BUFFER_FORMAT_YUV420_10BIT, "P010" },
        { NV_ENC_BUFFER_FORMAT_YUV444_10BIT, "YUV444P10" },
        { NV_ENC_BUFFER_FORMAT_ARGB,         "ARGB" },
        { NV_ENC_BUFFER_FORMAT_ARGB10,       "ARGB10" },
        { NV_ENC_BUFFER_FORMAT_AYUV,         "AYUV" },
        { NV_ENC_BUFFER_FORMAT_ABGR,         "ABGR" },
        { NV_ENC_BUFFER_FORMAT_ABGR10,       "ABGR10" },
};

static const struct {
    const GUID *guid;
    const char *desc;
} nvenc_profiles[] = {
        { &NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID,        "Auto" },
        { &NV_ENC_H264_PROFILE_BASELINE_GUID,           "Baseline" },
        { &NV_ENC_H264_PROFILE_MAIN_GUID,               "Main" },
        { &NV_ENC_H264_PROFILE_HIGH_GUID,               "High" },
        { &NV_ENC_H264_PROFILE_HIGH_444_GUID,           "High444" },
        { &NV_ENC_H264_PROFILE_STEREO_GUID,             "MVC" },
        { &NV_ENC_H264_PROFILE_SVC_TEMPORAL_SCALABILTY, "SVC" },
        { &NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID,   "Progressive High" },
        { &NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID,   "Constrained High" },
        { &NV_ENC_HEVC_PROFILE_MAIN_GUID,               "Main" },
        { &NV_ENC_HEVC_PROFILE_MAIN10_GUID,             "Main10" },
        { &NV_ENC_HEVC_PROFILE_FREXT_GUID,              "Main444" },
};


static const struct {
    const GUID *guid;
    const char *desc;
} nvenc_presets[] = {
        { &NV_ENC_PRESET_DEFAULT_GUID,             "default" },
        { &NV_ENC_PRESET_HP_GUID,                  "hp"},
        { &NV_ENC_PRESET_HQ_GUID,                  "hq"},
        { &NV_ENC_PRESET_BD_GUID,                  "bluray"},
        { &NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID, "ll" },
        { &NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,      "llhq" },
        { &NV_ENC_PRESET_LOW_LATENCY_HP_GUID,      "llhp" },
        { &NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,    "lossless" },
        { &NV_ENC_PRESET_LOSSLESS_HP_GUID,         "losslesshp" },
};


#define NVENCAPI_CHECK_VERSION(major, minor) \
    ((major) < NVENCAPI_MAJOR_VERSION || ((major) == NVENCAPI_MAJOR_VERSION && (minor) <= NVENCAPI_MINOR_VERSION))

static void nvenc_get_driver_requirement(int &major, int& minor)
{
#if NVENCAPI_CHECK_VERSION(9, 1)
    # if defined(_WIN32) || defined(__CYGWIN__)
        major = 436;
        minor = 15;
# else
        major = 435;
        minor = 21;
# endif
#elif NVENCAPI_CHECK_VERSION(8, 1)
# if defined(_WIN32) || defined(__CYGWIN__)
        major = 390;
        minor = 77;
# else
    major = 390;
    minor = 25;
# endif
#else
    # if defined(_WIN32) || defined(__CYGWIN__)
        major = 378;
        minor = 66;
# else
        major = 378;
        minor = 13;
# endif
#endif
}

inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
            {0x30, 192},
            {0x32, 192},
            {0x35, 192},
            {0x37, 192},
            {0x50, 128},
            {0x52, 128},
            {0x53, 128},
            {0x60,  64},
            {0x61, 128},
            {0x62, 128},
            {0x70,  64},
            {0x72,  64},
            {0x75,  64},
            {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    LOG(WARNING) << "MapSMtoCores for SM " << major << "." << minor << " is undefined. Default to use %d " << nGpuArchCoresPerSM[index - 1].Cores << "Cores/SM";
    return nGpuArchCoresPerSM[index - 1].Cores;
}


bool DeviceInfo::getDeviceInfo(int deviceId, DeviceInfoT& deviceInfo) {

    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    DeviceInfoT di;
    di.device_name = std::string(deviceProp.name);

    int driverVersion = 0, runtimeVersion = 0;

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    di.driver_major = driverVersion / 1000;
    di.driver_minor = (driverVersion % 100) / 10;
    di.runtime_major = runtimeVersion / 1000;
    di.runtime_minor = (runtimeVersion % 100) / 10;
    di.capability_major = deviceProp.major;
    di.capability_minor = deviceProp.minor;
    di.count_multiprocessors = deviceProp.multiProcessorCount;
    di.count_cudacores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    di.total_memory_bytes = (unsigned long long)deviceProp.totalGlobalMem;

    uint32_t nvenc_max_ver;
    CHECK_NV(NvEncodeAPIGetMaxSupportedVersion(&nvenc_max_ver));
    di.nvenc_major = nvenc_max_ver >> 4;
    di.nvenc_minor = nvenc_max_ver & 0xf;

    // check if driver is compatible with nvidia-codec-sdk
    int req_driver_major = 0, req_driver_minor = 0;
    nvenc_get_driver_requirement(req_driver_major, req_driver_minor);
    di.required_driver_major = req_driver_major;
    di.required_driver_minor = req_driver_minor;

    CUcontext cuda_ctx;
    CUcontext dummy;

    // decoder info
    ck(cuCtxCreate(&cuda_ctx, CU_CTX_SCHED_BLOCKING_SYNC, deviceId));
    for (int c = 0; c < cudaVideoCodec_NumCodecs; c++) {
        for (int f = 0; f < 4; f++) {
            for (int b = 8; b < 14; b += 2) {
                di.decoder_capabilities.push_back(get_decoder_caps((cudaVideoCodec)c, (cudaVideoChromaFormat)f, b));
            }
        }
    }
    cuCtxPopCurrent(&dummy);





    deviceInfo = di;
}

NVdecInfoT DeviceInfo::get_decoder_caps(cudaVideoCodec codec_type, cudaVideoChromaFormat chroma_format, unsigned int bit_depth) {
    NVdecInfoT info;

    switch (codec_type) {
        case cudaVideoCodec_MPEG1:
            info.codec_type = CodecType::MPEG1;
            break;
        case cudaVideoCodec_MPEG2:
            info.codec_type = CodecType::MPEG2;
            break;
        case cudaVideoCodec_MPEG4:
            info.codec_type = CodecType::MPEG4;
            break;
        case cudaVideoCodec_VC1:
            info.codec_type = CodecType::VC1;
            break;
        case cudaVideoCodec_H264:
            info.codec_type = CodecType::H264;
            break;
        case cudaVideoCodec_JPEG:
            info.codec_type = CodecType::MJPEG;
            break;
        case cudaVideoCodec_H264_SVC:
            info.codec_type = CodecType::H264_SVC;
            break;
        case cudaVideoCodec_H264_MVC:
            info.codec_type = CodecType::H264_MVC;
            break;
        case cudaVideoCodec_HEVC:
            info.codec_type = CodecType::HEVC;
            break;
        case cudaVideoCodec_VP8:
            info.codec_type = CodecType::VP8;
            break;
        case cudaVideoCodec_VP9:
            info.codec_type = CodecType::VP9;
            break;
        default:
            info.codec_type = CodecType::UNKNOWN;
            break;
    }

    switch (chroma_format) {
        case cudaVideoChromaFormat_Monochrome:
            info.chroma_format = ChromaFormatType::YUV400;
            break;
        case cudaVideoChromaFormat_420:
            info.chroma_format = ChromaFormatType::YUV420;
            break;
        case cudaVideoChromaFormat_422:
            info.chroma_format = ChromaFormatType::YUV422;
            break;
        case cudaVideoChromaFormat_444:
            info.chroma_format = ChromaFormatType::YUV444;
            break;
    }

    CUVIDDECODECAPS caps;
    caps.eCodecType = codec_type;
    caps.eChromaFormat = chroma_format;
    caps.nBitDepthMinus8 = bit_depth - 8;

    ck(cuvidGetDecoderCaps(&caps));

    if (!caps.bIsSupported) {
        info.is_supported = false;
        return info;
    }

    switch (caps.nOutputFormatMask) {
        case 1 << cudaVideoSurfaceFormat_NV12:
            info.video_surface = VideoSurfaceFormatType::NV12;
            break;
        case 1 << cudaVideoSurfaceFormat_P016:
            info.video_surface = VideoSurfaceFormatType::P016;
            break;
        case 1 << cudaVideoSurfaceFormat_YUV444:
            info.video_surface = VideoSurfaceFormatType::YUV444P;
            break;
        case 1 << cudaVideoSurfaceFormat_YUV444_16Bit:
            info.video_surface = VideoSurfaceFormatType::YUV444P16;
            break;
        case (1 << cudaVideoSurfaceFormat_NV12) + (1 << cudaVideoSurfaceFormat_P016):
            info.video_surface = VideoSurfaceFormatType::P016_NV12;
            break;
        default:
            info.video_surface = VideoSurfaceFormatType::UNKNOWN;
            break;
    }    

    info.bit_depth = bit_depth;
    info.max_width = caps.nMaxWidth;
    info.max_height = caps.nMaxHeight;
    info.is_supported = true;

    return info;
}


bool DeviceInfo::deviceCount(int& count) {
    int deviceCount = 0;
    if (!ck(cudaGetDeviceCount(&deviceCount))) {
        count = 0;
        return false;
    }
    count = deviceCount;
    return true;
}