#include "lightglue/lightglue.h"

#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>

namespace py = pybind11;
using lightglue::FeatureExtractor;

FeatureExtractor::FeatureExtractor(int max_keypoints, float detection_threshold, int nms_radius)
{
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    py::module_ feature_module = py::module_::import("python.binding");
    py_extractor = feature_module.attr("FeatureExtractor")(max_keypoints, detection_threshold, nms_radius);
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat>
FeatureExtractor::extract_features(const cv::Mat& image)
{
    // Convert to py array
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
    py::array_t<float> scores = result.cast<py::tuple>()[2].cast<py::array_t<float>>();

    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < kpts.shape(0); ++i) {
        keypoints.push_back(cv::KeyPoint(kpts.at(i, 0), kpts.at(i, 1), 1, -1, scores.at(i)));
    }

    cv::Mat descriptors(desc.shape(0), desc.shape(1), CV_32F, const_cast<float*>(desc.data()));
    return std::make_tuple(keypoints, descriptors.clone());
}

static py::array_t<float> to_py_array(const cv::Mat& mat)
{
    std::vector<size_t> shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)};
    std::vector<size_t> strides = {static_cast<size_t>(mat.step[0]),
                                   static_cast<size_t>(mat.step[1])};
    return py::array_t<float>(shape, strides, (float*)mat.data);
}

static py::array_t<float> to_py_array(const std::vector<cv::KeyPoint>& kps)
{
    std::vector<float> data(kps.size() * 2);
    for (int i = 0; i < kps.size(); ++i) {
        data[i * 2] = kps[i].pt.x;
        data[i * 2 + 1] = kps[i].pt.y;
    }
    std::vector<size_t> shape = {static_cast<size_t>(kps.size()), 2};
    std::vector<size_t> strides = {sizeof(float) * 2, sizeof(float)};
    return py::array_t<float>(shape, strides, data.data());
}

std::vector<cv::DMatch> FeatureExtractor::match_features(const std::vector<cv::KeyPoint>& kps1,
                                                         const std::vector<cv::KeyPoint>& kps2,
                                                         const cv::Mat& desc1,
                                                         const cv::Mat& desc2,
                                                         const cv::Size& img_size1,
                                                         const cv::Size& img_size2)
{
    // Convert to py array
    py::array_t<float> kps1_array = to_py_array(kps1);
    py::array_t<float> kps2_array = to_py_array(kps2);
    py::array_t<float> desc1_array = to_py_array(desc1);
    py::array_t<float> desc2_array = to_py_array(desc2);

    // Python
    auto result = py_extractor.attr("match_features")(kps1_array,
                                                      kps2_array,
                                                      desc1_array,
                                                      desc2_array,
                                                      img_size1.width,
                                                      img_size1.height,
                                                      img_size2.width,
                                                      img_size2.height);

    // Convert to OpenCV
    py::array_t<int> matches = result.cast<py::tuple>()[0].cast<py::array_t<int>>();
    py::array_t<float> scores = result.cast<py::tuple>()[1].cast<py::array_t<float>>();

    std::vector<cv::DMatch> dmatches;
    for (int i = 0; i < matches.shape(0); ++i) {
        dmatches.push_back(cv::DMatch(matches.at(i, 1), matches.at(i, 0), scores.at(i)));
    }
    return dmatches;
}
