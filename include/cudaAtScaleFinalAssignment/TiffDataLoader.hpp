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
            std::filesystem::directory_iterator dirIt(_pathToData);
            for (auto const &dir_entry : dirIt) {
                _imageNames.push_back(dir_entry.path());
            }
            _numImages = _imageNames.size();
        }

        std::string getNextImage() {
            if (_imageCounter < _numImages) {
                auto ret = _imageNames.at(_imageCounter);
                _imageCounter++;
                return ret;
            } else {
                return "All Images finished";
            }
        }

    private:
        std::string _pathToData;
        std::vector<std::string> _imageNames;
        std::size_t _numImages;
        std::size_t _imageCounter{0};
};