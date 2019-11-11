//
// Created by netlabs on 11/9/19.
//

#ifndef NVPIPE_DEVICEINFO_H
#define NVPIPE_DEVICEINFO_H

#include "NvPipe/NvCodec/NvEncoder/NvEncoder.h"
#include "NvPipe/NvCodec/NvDecoder/NvDecoder.h"


enum class CodecType {
    UNKNOWN = 0,
    MPEG1,
    MPEG2,
    MPEG4,
    VC1,
    H264,
    MJPEG,
    H264_SVC,
    H264_MVC,
    HEVC,
    VP8,
    VP9
};

enum class ChromaFormatType {
    UNKNOWN = 0,
    YUV400,
    YUV420,
    YUV422,
    YUV444
};

enum class VideoSurfaceFormatType {
    UNKNOWN = 0,
    NV12,
    P016,
    YUV444P,
    YUV444P16,
    P016_NV12
};

struct NVencInfoT {

};

struct NVdecInfoT {
    bool is_supported{false};
    CodecType codec_type{CodecType ::UNKNOWN};
    ChromaFormatType chroma_format{ChromaFormatType ::UNKNOWN};
    VideoSurfaceFormatType video_surface{VideoSurfaceFormatType ::UNKNOWN};
    int max_width{0};
    int max_height{0};
    int bit_depth{0};
};

struct DeviceInfoT {
    std::string device_name;
    int driver_major{0};
    int driver_minor{0};
    int required_driver_major{0};
    int required_driver_minor{0};
    int runtime_major{0};
    int runtime_minor{0};
    int capability_major{0};
    int capability_minor{0};
    int nvenc_major{0};
    int nvenc_minor{0};

    int count_multiprocessors{0};
    int count_cudacores{0};
    unsigned long long total_memory_bytes{0};



    std::vector<NVdecInfoT> decoder_capabilities;
};


class DeviceInfo {

public:
    DeviceInfo() = default;

    bool getDeviceInfo(int deviceId, DeviceInfoT& deviceInfo);

    bool deviceCount(int& count);

protected:
    NVdecInfoT get_decoder_caps(cudaVideoCodec codec_type, cudaVideoChromaFormat chroma_format, unsigned int bit_depth);

};


#endif //NVPIPE_DEVICEINFO_H
