#include <iostream>
#include <string>
#include "CLI11.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

#include "cudaAtScaleFinalAssignment/ImageTransformation.hpp"
#include "cudaAtScaleFinalAssignment/TiffDataLoader.hpp"

#include <fstream>

int main(int argc, char** argv) {
    // command line parsing
    CLI::App app{"App description"};
    argv = app.ensure_utf8(argv);
    std::string pathToData = "default";
    app.add_option("-p,--path", pathToData, "Path to the tiff data. We only want the path. The program will iterate over all *.tiff data within this folder (in a non-recursive manner!).");
    CLI11_PARSE(app, argc, argv);

    // data input
    TiffDataLoader dataLoader(pathToData);

    // prepare GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA capable devices found!" << std::endl;
        return 1;
    }
    std::cout << "Device count is: " << deviceCount << std::endl;
    cudaSetDevice(0);

    // main loop
    std::cout << "Processing " << dataLoader.getNumImages() << " images" << std::endl;
    bool terminate = false;
    while (!terminate) {
        auto tmpImg = dataLoader.getNextImage();
        if ("" == tmpImg || "Error" == tmpImg) {
            std::cout << "DEBUG Termination" << std::endl;
            terminate = true;
            continue;
        }
        std::cout << "Processing image size: " << tmpImg << std::endl;
        npp::ImageCPU_8u_C1 oHostSrc;
        npp::loadImage(tmpImg, oHostSrc);
    }

    /// TESTING CODE
    ImageTransformation imgTrans;
    
    return EXIT_SUCCESS;
}