#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <memory>
#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

class PgmDataGetter {
    public:
        PgmDataGetter(const std::string &pathToFolder) :
            _pathToData{pathToFolder}
        {
            std::cout << "Scanning for data..." << std::endl;
            std::filesystem::directory_iterator dirIt(_pathToData);
            for (auto const &dir_entry : dirIt) {
                if (endsWith(dir_entry.path(), ".pgm")) {
                    _imageNames.push_back(dir_entry.path());
                } else {
                    std::cerr <<"The file " << dir_entry.path() << " is no png image - it will be ignored by the processing" << std::endl;
                }
            }
            _numImages = _imageNames.size();
        }

        std::string getNextImage()
        {
            if (_imageCounter < _numImages)
            {
                auto ret = _imageNames.at(_imageCounter);
                _imageCounter++;

                if (isImageValid(ret))
                    return ret;
                else {
                    return "Error";
                }
            }
            else
            {
                return "";
            }
        }

        std::size_t getNumImages() const {
            return _numImages;
        }

    private:
        bool endsWith(const std::string &str, const std::string &suffix)
        {
            return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool isImageValid(const std::string& path) {
            int file_errors = 0;
            std::ifstream infile(path.data(), std::ifstream::in);
            if (infile.good())
            {
                file_errors = 0;
                infile.close();
            }
            else
            {
                file_errors++;
                infile.close();
            }
            if (file_errors > 0)
            {
                return false;
            } else {
                return true;
            }
        }

        std::string _pathToData;
        std::vector<std::string> _imageNames;
        std::size_t _numImages;
        std::size_t _imageCounter{0};
};