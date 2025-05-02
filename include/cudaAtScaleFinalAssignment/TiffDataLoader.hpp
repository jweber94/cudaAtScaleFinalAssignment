#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

class TiffDataLoader {
    public:
        TiffDataLoader(const std::string &pathToFolder) :
            _pathToData{pathToFolder}
        {
            std::cout << "Scanning for data..." << std::endl;
            std::filesystem::directory_iterator dirIt(_pathToData);
            for (auto const &dir_entry : dirIt) {
                if (endsWith(dir_entry.path(), ".tiff")) {
                    _imageNames.push_back(dir_entry.path());
                } else {
                    std::cerr <<"The file " << dir_entry.path() << " is no tiff image - it will be ignored by the processing" << std::endl;
                }
            }
            _numImages = _imageNames.size();
        }

        std::string getNextImage() {
            if (_imageCounter < _numImages)
            {
                auto ret = _imageNames.at(_imageCounter);
                _imageCounter++;
                return ret;
            }
            else
            {
                return "All Images finished";
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

        std::string _pathToData;
        std::vector<std::string> _imageNames;
        std::size_t _numImages;
        std::size_t _imageCounter{0};
};