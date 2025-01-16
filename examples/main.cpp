#include <opencv2/opencv.hpp>
#include <lightglue.h>

int main() {
    cv::Mat img = cv::imread("assets/frame0.jpg");

    // Feature extraction
    lightglue::FeatureExtractor extractor;
    auto [keypoints, descriptors] = extractor.extract_features(img);
    std::cout << "Keypoints: " << keypoints.size() << std::endl;
    std::cout << "Descriptors: " << descriptors.size() << std::endl;

    return 0;
}