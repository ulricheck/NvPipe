//
// Created by narvis on 11/11/19.
//

#include "NvPipe/Utils/DeviceInfo.h"
#include "NvPipe/Utils/NvCodecUtils.h"
#include "spdlog/spdlog.h"

std::string codec2string(CodecType ct) {
    switch (ct) {
        case CodecType::UNKNOWN:
            return "UNKNOWN";
        case CodecType::MPEG1:
            return "MPEG1";
        case CodecType::MPEG2:
            return "MPEG2";
        case CodecType::MPEG4:
            return "MPEG4";
        case CodecType::VC1:
            return "VC1";
        case CodecType::H264:
            return "H264";
        case CodecType::MJPEG:
            return "MJPEG";
        case CodecType::H264_SVC:
            return "H264_SVC";
        case CodecType::H264_MVC:
            return "H264_MVC";
        case CodecType::HEVC:
            return "HEVC";
        case CodecType::VP8:
            return "VP8";
        case CodecType::VP9:
            return "VP9";
        default:
            return "UNKNOWN";
    }
}


int main(int argc, char *argv[])
{
    ck(cuInit(0));
    auto di = DeviceInfo();

    int numDevices{0};
    if (!di.deviceCount(numDevices)) {
        spdlog::error("Error while getting GPU device count.");
        return 1;
    }

    if (numDevices == 0) {
        spdlog::error("No nvidia GPU's found.");
        return 1;
    }

    DeviceInfoT device_info;
    for (int i=0; i < numDevices; i++) {
        di.getDeviceInfo(i, device_info);
        spdlog::info("Found device: {0} with {1} CUDA cores", device_info.device_name, device_info.count_cudacores);
        spdlog::info("Minimum device driver needed: {0}.{1}", device_info.required_driver_major, device_info.required_driver_minor);
        for (auto& dec_cap : device_info.decoder_capabilities) {
            if (dec_cap.is_supported) {
                spdlog::info("  Supported Decoder: {0} maxWidth: {1} maxHeight: {2}", codec2string(dec_cap.codec_type), dec_cap.max_width, dec_cap.max_height);
            }
        }
    }
}