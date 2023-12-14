#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

int main(int, char**){
    // read input image
cv::Mat image = cv::imread("C:/Users/denis/Desktop/CodeProjects/opencv/signal-2023-12-14-212155_003.jpeg",cv::IMREAD_GRAYSCALE );
if (image.empty()) {
        std::cout << "Error opening image file\n";
        return -1;
    }
cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);

    // Draw keypoints on the image
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    // Display the original image and the image with keypoints
    cv::imshow("Original Image", image);
    cv::imshow("FAST Features", output);
    cv::waitKey(0);

    cv::destroyAllWindows();
    std::getchar();
    return 0;
    
}
