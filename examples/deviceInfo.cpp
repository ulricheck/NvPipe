//
// Created by narvis on 11/11/19.
//

#include "NvPipe/Utils/DeviceInfo.h"
#include "NvPipe/Utils/NvCodecUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

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
        LOG(ERROR) << "Error while getting GPU device count.";
        return 1;
    }

    if (numDevices == 0) {
        LOG(ERROR) << "No nvidia GPU's found.";
        return 1;
    }

    DeviceInfoT device_info;
    for (int i=0; i < numDevices; i++) {
        di.getDeviceInfo(i, device_info);
        LOG(INFO) << "Found device: " << device_info.device_name << " with " << device_info.count_cudacores << " CUDA cores" ;
        LOG(INFO) << "Minimum device driver needed: " << device_info.required_driver_major << "." << device_info.required_driver_minor;
        for (auto& dec_cap : device_info.decoder_capabilities) {
            if (dec_cap.is_supported) {
                LOG(INFO) << "  Supported Decoder: " << codec2string(dec_cap.codec_type) << " maxWidth: " << dec_cap.max_width << " maxHeight: " << dec_cap.max_height;
            }
        }
    }
}