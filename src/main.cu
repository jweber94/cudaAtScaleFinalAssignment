#include <iostream>
#include "CLI11.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

#include "cudaAtScaleFinalAssignment/ImageTransformation.hpp"


int main(int argc, char** argv) {
    std::cout << "Hello World" << std::endl;
    ImageTransformation imgTrans;
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA capable devices found!" << std::endl;
        return 1;
    }
    std::cout << "Device count is: " << deviceCount << std::endl;
    cudaSetDevice(0);

    npp::ImageCPU_8u_C1 oHostSrc;
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
 
     // create struct with box-filter mask size
     NppiSize oMaskSize = {5, 5};
 
     NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
     NppiPoint oSrcOffset = {0, 0};
 
     // create struct with ROI size
     NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
     // allocate device image of appropriately reduced size
     npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    return EXIT_SUCCESS;
}