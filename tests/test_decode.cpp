/* **********************************************************************************
#                                                                                   #
# Copyright (c) 2019,                                                               #
# Research group CAMP                                                               #
# Technical University of Munich                                                    #
#                                                                                   #
# All rights reserved.                                                              #
# Ulrich Eck - ulrich.eck@tum.de                                                    #
#                                                                                   #
# Redistribution and use in source and binary forms, with or without                #
# modification, are restricted to the following conditions:                         #
#                                                                                   #
#  * The software is permitted to be used internally only by the research group     #
#    CAMP and any associated/collaborating groups and/or individuals.               #
#  * The software is provided for your internal use only and you may                #
#    not sell, rent, lease or sublicense the software to any other entity           #
#    without specific prior written permission.                                     #
#    You acknowledge that the software in source form remains a confidential        #
#    trade secret of the research group CAMP and therefore you agree not to         #
#    attempt to reverse-engineer, decompile, disassemble, or otherwise develop      #
#    source code for the software or knowingly allow others to do so.               #
#  * Redistributions of source code must retain the above copyright notice,         #
#    this list of conditions and the following disclaimer.                          #
#  * Redistributions in binary form must reproduce the above copyright notice,      #
#    this list of conditions and the following disclaimer in the documentation      #
#    and/or other materials provided with the distribution.                         #
#  * Neither the name of the research group CAMP nor the names of its               #
#    contributors may be used to endorse or promote products derived from this      #
#    software without specific prior written permission.                            #
#                                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND       #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      #
#                                                                                   #
*************************************************************************************/

#include <fstream>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

#include "NvPipe/NvPipe.h"
#include <cuda_runtime_api.h>

inline bool isDevicePointer(const void* ptr)
{
    struct cudaPointerAttributes attr;
    const cudaError_t perr = cudaPointerGetAttributes(&attr, ptr);
    return (perr == cudaSuccess) && (attr.type != cudaMemoryTypeHost);
}

TEST(NvencTestSuite, SimpleDecodeTest) {
    spdlog::set_level(spdlog::level::trace);

// reads example stream from nvpipe nvExampleFile (stream.bin)
    const uint32_t width = 3840;
    const uint32_t height = 2160;
    const uint32_t bytesPerPixel = 4;
    std::vector<uint8_t> compressed(width * height * bytesPerPixel);

    std::string fname("../tests/stream.bin");

    spdlog::info("Setup Decoder");
    auto decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, NVPIPE_H264, width, height);
    ASSERT_FALSE(!decoder);

    uint32_t dataSize = width * height * bytesPerPixel;
    uint8_t* frameBuffer = (uint8_t *)malloc(dataSize);
    void* gpuBuffer{nullptr};
    cudaMalloc(&gpuBuffer, dataSize);
    ASSERT_TRUE(isDevicePointer(gpuBuffer));

    spdlog::info("Enter Mainloop.");
    for (size_t n=0; n<10; n++) {

        std::ifstream in(fname, std::ios::in | std::ios::binary);
        ASSERT_FALSE(!in );
        if (!in) {
            spdlog::error("Error: Failed to open input file: {0}", fname);
            return;
        }

        for (uint32_t i = 0; i < 10; ++i) {
            uint64_t size;
            in.read((char *) &size, sizeof(uint64_t));
            in.read((char *) compressed.data(), size);

            uint64_t decoded_size = NvPipe_Decode(decoder, compressed.data(), size, gpuBuffer, width, height);
            ASSERT_TRUE(decoded_size > 0);

        }

        in.close();
    }
    cudaFree(gpuBuffer);
    NvPipe_Destroy(decoder);

    spdlog::info("Finished test");
}


