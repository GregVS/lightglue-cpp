#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>

#include "lightglue.h"

namespace py = pybind11;
using lightglue::FeatureExtractor;

FeatureExtractor::FeatureExtractor(int max_keypoints)
{
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    py::module_ feature_module = py::module_::import("python.binding");
    py_extractor = feature_module.attr("FeatureExtractor")(max_keypoints);
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> FeatureExtractor::extract_features(const cv::Mat& image)
{
    // Convert to numpy array
    std::vector<size_t> shape = {static_cast<size_t>(image.rows), 
                                static_cast<size_t>(image.cols), 
                                static_cast<size_t>(image.channels())};
    std::vector<size_t> strides = {static_cast<size_t>(image.step[0]), 
                                  static_cast<size_t>(image.step[1]), 
                                  static_cast<size_t>(1)};
    py::array_t<unsigned char> img_array(shape, strides, image.data);

    // Python
    auto result = py_extractor.attr("extract_features")(img_array);
    
    // Convert to OpenCV image
    py::array_t<float> kpts = result.cast<py::tuple>()[0].cast<py::array_t<float>>();
    py::array_t<float> desc = result.cast<py::tuple>()[1].cast<py::array_t<float>>();
    
    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < kpts.shape(0); ++i) {
        keypoints.push_back(cv::KeyPoint(kpts.at(i, 0), kpts.at(i, 1), 1));
    }
    
    cv::Mat descriptors(desc.shape(0), desc.shape(1), CV_32F,
                       const_cast<float*>(desc.data()));
    
    return std::make_tuple(keypoints, descriptors);
} 