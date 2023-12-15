#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#define THRESHOLD 20 // Threshold for corner detection
bool isCorner(const cv::Mat& image, int x, int y) {
    uchar center = image.at<uchar>(y, x);
    int pixelOffsets[16][2] = { //16 pixels with x/y coordinates
        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3},
        {0, 3},  {1, 3},  {2, 2},  {3, 1},
        {3, 0},  {3, -1}, {2, -2}, {1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
    };

    int count = 0;
    for (int i = 0; i < 16; ++i) {
        int offsetX = x + pixelOffsets[i][0];
        int offsetY = y + pixelOffsets[i][1];

        if (offsetX >= 0 && offsetX < image.cols && offsetY >= 0 && offsetY < image.rows) {
            uchar currentPixel = image.at<uchar>(offsetY, offsetX);
            if (center + THRESHOLD <= currentPixel || center - THRESHOLD >= currentPixel)
                count++;
        }
    }

    return count >= 12; // If 12 or more pixels out of 16 are radically brighter or darker than the center pixel
    //12 is magic number from creator of this algorithm
}
std::vector<cv::Point> detectCorners(const cv::Mat& image) {
    std::vector<cv::Point> corners;
    for (int y = 3; y < image.rows - 3; ++y) {
        for (int x = 3; x < image.cols - 3; ++x) {
            if (isCorner(image, x, y))
                corners.emplace_back(cv::Point(x, y));
        }
    }
    return corners;
}

int main(int, char**){
    // read input image
    cv::Mat image = cv::imread("C:/Users/denis/Desktop/CodeProjects/opencv/003.jpeg",cv::IMREAD_GRAYSCALE );
    if (image.empty()) {
            std::cout << "Error opening image file\n";
            return -1;
        }
    
    std::vector<cv::Point> corners = detectCorners(image);
    

    cv::Mat result = image.clone();
    for (const auto& corner : corners) {
        circle(result, corner, 1, cv::Scalar(255, 0, 0), 2); //scalar is also a grayscale
    }
    
        
    //imshow("Original Image", image);
    imshow("FAST Corner Detection", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
     
}
