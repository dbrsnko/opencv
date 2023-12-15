#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

int threshold = 20; // Threshold for corner detection
bool isCorner(const cv::Mat& image, int x, int y) { //FAST evaluation
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
            if (center + threshold <= currentPixel || center - threshold>= currentPixel)
                count++;
        }
    }

    return count >= 12; // If 12 or more pixels out of 16 are radically brighter or darker than the center pixel
    //12 is magic number from creator of this algorithm
}
std::vector<cv::Point> detectCorners(const cv::Mat& image) { //FAST detection
    std::vector<cv::Point> corners;
    for (int y = 3; y < image.rows - 3; ++y) {
        for (int x = 3; x < image.cols - 3; ++x) {
            if (isCorner(image, x, y))
                corners.emplace_back(cv::Point(x, y));
        }
    }
    return corners;
}
//thomasi parameters
int tomasi_qualityLevel = 1; //multiplier of minimal accepted quality of image corners, less value produces more corners( changing type to double is ok too)
int max_qualityLevel = 3; //diviser of minimal accepted quality of image corners, more value produces more corners


cv::RNG rng(12345); //for colorful circles
void tomasi_function(cv::Mat image,int blockSize, int apertureSize){ //tomasi alg, provide color image; 
//block size - for each pixel it calculates a 2x2 gradient covariance matrix over a size x size neighbourhood
//aperture is for sobel operator, better use 3,5 or 7
    cv::Mat tomasi_copy, image_grey, tomasi_dst; //tomasi_copy stores copy of image, image_grey stores greyscale tomasi_copy, tomasi_dst stores minimal eigenvalues
    tomasi_copy = image.clone();
    cv::cvtColor(image,image_grey, cv::COLOR_BGR2GRAY);
    cv::cornerMinEigenVal( image_grey, tomasi_dst, blockSize, apertureSize );//Calculates the minimal eigenvalue of gradient matrices for corner detection.
    double tomasi_minVal, tomasi_maxVal; 
    minMaxLoc( tomasi_dst, &tomasi_minVal, &tomasi_maxVal );//finds the minimum and maximum element values and their positions
    tomasi_qualityLevel = MAX(tomasi_qualityLevel, 1);
    for( int i = 0; i < image_grey.rows; i++ )
    {
        for( int j = 0; j < image_grey.cols; j++ )
        {
            if( tomasi_dst.at<float>(i,j) > tomasi_minVal + ( tomasi_maxVal - tomasi_minVal )*tomasi_qualityLevel/max_qualityLevel ) //if point got enough value then print it
            {
                circle( tomasi_copy, cv::Point(j,i), 4, cv::Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), cv::FILLED );
            }
        }
    }
    imshow( "Tomasi", tomasi_copy );
    //TODO make a return that provides vector with points to further pass it to Lucas-Kanade
}

int main(int, char**){
    std::string gifFileName = "C:/Users/denis/Desktop/CodeProjects/opencv/4.gif"; // Replace with your GIF file name

    cv::VideoCapture cap(gifFileName);

    if (!cap.isOpened()) {
        std::cout << "Error opening file!" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    cv::Mat past_frame;
    int frameToRead = 0;
    int n=10; // read each n frame, 10-15 frames recommended
    cap.set(cv::CAP_PROP_POS_FRAMES, frameToRead); //make it since we want a past_frame to be present
    cap >> frame;

    
    while (!frame.empty()){
        past_frame = frame.clone(); 
        frameToRead+=n;
        cap.set(cv::CAP_PROP_POS_FRAMES, frameToRead);
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Frame empty" << frameToRead << std::endl;
            break;
        }
        
        cv::Mat grey;
        cv::cvtColor(frame,grey, cv::COLOR_BGR2GRAY); //convert frame to grayscale since fast operates in grayscale*/
        tomasi_function(frame,3,5);

        /*
        std::vector<cv::Point> corners = detectCorners(gray);
        for (const auto& corner : corners)  //draw keypoints on colored image
            circle(frame, corner, 1, cv::Scalar(255, 0, 0), 2); 
        cv::imshow("Past Frame", past_frame);*/
        cv::imshow("Frame", frame);
   


    cv::waitKey(0);
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
     
    /* FAST implementation
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
     */

}
