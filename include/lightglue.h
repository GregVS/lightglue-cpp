#pragma once

#include <tuple>
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>

namespace lightglue 
{

class FeatureExtractor {
public:
    FeatureExtractor(int max_keypoints = 1024);
    
    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract_features(const cv::Mat& image);

private:
    pybind11::object py_extractor;
}; 

} // namespace lightglue