#pragma once

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>

#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

class ImageProcessor {
    public:
        ImageProcessor(const std::string& processingOutputFolder) :
            m_outputFolder{processingOutputFolder}
        {
            // check if GPU exists to run the algorithm on it
            cudaGetDeviceCount(&m_deviceCount);
            if (m_deviceCount == 0)
            {
                std::cerr << "No CUDA capable devices found! The program can not work without a GPU" << std::endl;
                exit(EXIT_FAILURE);
            }
            std::cout << "Devices Found - count is: " << m_deviceCount << " using device No. 0" << std::endl;
            cudaSetDevice(0);
        }

        bool processImage(const std::string& pathToPgm) {
            std::cout << "DEBUG: Processing image size: " << pathToPgm << std::endl;
            npp::ImageCPU_8u_C1 oHostSrc;
            npp::loadImage(pathToPgm, oHostSrc);
            return true;
        }

        ImageProcessor() = delete;

    private:
        std::string m_outputFolder;
        int m_deviceCount{0};
};