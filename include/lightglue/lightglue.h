#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace lightglue {

class FeatureExtractor {
  public:
    FeatureExtractor(int max_keypoints = 1024, float detection_threshold = 0.0005, int nms_radius=4);

    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract_features(const cv::Mat& image);

    std::vector<cv::DMatch> match_features(const std::vector<cv::KeyPoint>& kps1,
                                           const std::vector<cv::KeyPoint>& kps2,
                                           const cv::Mat& desc1,
                                           const cv::Mat& desc2,
                                           const cv::Size& img_size1,
                                           const cv::Size& img_size2);

  private:
    pybind11::object py_extractor;
};

} // namespace lightglue