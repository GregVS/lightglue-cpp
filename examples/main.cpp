#include <lightglue/lightglue.h>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat frame0 = cv::imread("assets/frame0.png");
    cv::Mat frame1 = cv::imread("assets/frame1.png");

    // Feature extraction
    lightglue::FeatureExtractor extractor(2048, 0.05);
    auto [kps0, descs0] = extractor.extract_features(frame0);
    auto [kps1, descs1] = extractor.extract_features(frame1);

    // Feature matching
    auto matches = extractor
                       .match_features(kps0, kps1, descs0, descs1, frame0.size(), frame1.size());

    std::cout << "Matches: " << matches.size() << std::endl;

    cv::Mat left_frame = frame0.clone();
    cv::Mat right_frame = frame1.clone();
    for (const auto& match : matches) {
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        auto train_pt = kps0[match.trainIdx];
        auto score = train_pt.response;
        std::cout << "Score: " << score << std::endl;
        cv::circle(left_frame, train_pt.pt, 10 * score + 2, color, -1);

        auto query_pt = kps1[match.queryIdx];
        cv::circle(right_frame, query_pt.pt, 4, color, -1);
    }

    cv::Mat display_image = cv::Mat(left_frame.rows,
                                    left_frame.cols + right_frame.cols,
                                    left_frame.type());
    left_frame.copyTo(display_image(cv::Rect(0, 0, left_frame.cols, left_frame.rows)));
    right_frame.copyTo(
        display_image(cv::Rect(left_frame.cols, 0, right_frame.cols, right_frame.rows)));

    cv::imshow("Feature Matching", display_image);
    cv::waitKey(0);

    return 0;
}