from conans import ConanFile, CMake, tools
import os


class NvPipeConan(ConanFile):
    name = "nvpipe"
    version = "0.2"
    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"

    options = {
        "with_encoder": [True, False],
        "with_decoder": [True, False],
        "with_opengl": [True, False],
    }

    default_options = (
        "with_encoder=True",
        "with_decoder=True",
        "with_opengl=True",
        )

    exports = ["CMakeLists.txt", "FindNvPipe.cmake", "include/*", "src/*", "examples/*"]

    url="http://github.com/ulricheck/conan-nvpipe"
    license="nvidia demo code - license unknown"
    description="NVIDIA-accelerated zero latency video compression library for interactive remoting applications"
    
    requires = (
        "cuda_dev_config/[>=1.0]@camposs/stable",
        "nvidia-video-codec-sdk/9.1.23@vendor/stable",  # private repository due to license terms - just provides includes and libs
        )

    def build(self):
        """ Define your project building. You decide the way of building it
            to reuse it later in any other project.
        """
        cmake = CMake(self)

        cmake.definitions["NVPIPE_WITH_ENCODER"] = self.options.with_encoder
        cmake.definitions["NVPIPE_WITH_DECODER"] = self.options.with_decoder
        cmake.definitions["NVPIPE_WITH_OPENGL"] = self.options.with_opengl
        cmake.definitions["NVPIPE_BUILD_EXAMPLES"] = "OFF"

        cmake.configure()
        cmake.build()
        cmake.install()

    def package(self):
        """ Define your conan structure: headers, libs, bins and data. After building your
            project, this method is called to create a defined structure:
        """
        # Copy findZLIB.cmake to package
        self.copy("FindNvPipe.cmake", ".", ".")
        
    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
