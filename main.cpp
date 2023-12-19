#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>

int threshold = 20; // Threshold for FAST corner detection
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
float tomasi_qualityLevel = 4; //more value produces less corners, value should be between 0-1
int max_qualityLevel = 10; //divisor of minimal accepted quality of image corners, more value produces more corners

std::vector<cv::KeyPoint> tomasi(cv::Mat image_gray,int blockSize, int apertureSize){ //tomasi alg, provide grayscale image; 
//block size - for each pixel it calculates a 2x2 gradient covariance matrix over a size x size neighbourhood
//aperture is for sobel operator, better use 3,5 or 7
    std::vector<cv::KeyPoint> corners;
    cv::Mat tomasi_copy, tomasi_dst; //tomasi_copy stores copy of grayscale image, tomasi_dst stores minimal eigenvalues
    tomasi_copy = image_gray.clone();

    cv::cornerMinEigenVal( tomasi_copy, tomasi_dst, blockSize, apertureSize );//Calculates the minimal eigenvalue of gradient matrices for corner detection.

    double tomasi_minVal, tomasi_maxVal; 
    minMaxLoc( tomasi_dst, &tomasi_minVal, &tomasi_maxVal );//finds the minimum and maximum element values and their positions
    tomasi_qualityLevel = MAX(tomasi_qualityLevel, 0.5); //not less than 0,5 (for safety reasons)

    for( int i = 0; i < tomasi_copy.rows; i++ )
    {
        for( int j = 0; j < tomasi_copy.cols; j++ )
        {
            if( tomasi_dst.at<float>(i,j) > tomasi_minVal + ( tomasi_maxVal - tomasi_minVal )*tomasi_qualityLevel/max_qualityLevel ) //if point got enough value then print it
            {
                corners.emplace_back(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), 1, -1.0f);
                
            }
        }
    }
    
    return corners;
}




/// Optical flow tracker and interface
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const std::vector<cv::KeyPoint> &kp1_,
        std::vector<cv::KeyPoint> &kp2_,
        std::vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};


void OpticalFlowSingleLevel(
    const cv::Mat &img1, //the first image
    const cv::Mat &img2, //the second image
    const std::vector<cv::KeyPoint> &kp1, //keypoints in img1
    std::vector<cv::KeyPoint> &kp2, //kp2 keypoints in img2, if empty, use initial guess in kp1
    std::vector<bool> &success, //true if a keypoint is tracked successfully
    bool inverse = false, //use inverse formulation?
    bool has_initial_guess = false
);


void OpticalFlowMultiLevel( //multi level optical flow, scale of pyramid is set to 2 by default
    const cv::Mat &img1, //the first pyramid
    const cv::Mat &img2, //the second pyramid
    const std::vector<cv::KeyPoint> &kp1, //keypoints in img1
    std::vector<cv::KeyPoint> &kp2, //keypoints in img2
    std::vector<bool> &success, //true if a keypoint is tracked successfully
    bool inverse = false //set true to enable inverse formulation
);


inline float GetPixelValue(const cv::Mat &img, float x, float y) { //get a  value from reference gray scale image 
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}


int main(int argc, char **argv) {
    std::string gifFileName = "C:/Users/denis/Desktop/CodeProjects/opencv/4.gif"; // Replace with your GIF file name
    cv::VideoCapture cap(gifFileName);

    if (!cap.isOpened()) {
        std::cout << "Error opening file!" << std::endl;
        return -1;
    }
    cv::Mat img1, img2;   
    // Read the first frame
    int n = 1;//each n-th frame will be tracked 
    int frameToRead=0;  
    cap >> img1;
    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    cv::Mat mask = cv::Mat::zeros(img1.size(), img1.type());
    std::vector<cv::KeyPoint> kp1=tomasi(img1,2,3);        
    
    // now lets track these key points in the second image
    // first use single level LK in the validation picture

    while (1) {
        frameToRead+=n;
        cap.set(cv::CAP_PROP_POS_FRAMES, frameToRead);
        cap >> img2;
        if (img2.empty())
            break;
        
        cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);    
        std::vector<cv::KeyPoint> kp2;
        std::vector<bool> success_multi;
        
        OpticalFlowMultiLevel(img1, img2, kp1, kp2, success_multi, true);
        std::vector<cv::KeyPoint> goodNew;
        for (int i = 0; i < kp2.size(); i++) {
            if (success_multi[i]) {
                goodNew.push_back(kp2[i]);//store new good points for further tracking
                cv::circle(mask, kp2[i].pt, 2, cv::Scalar(255, 0, 0), 2);
                cv::line(mask, kp1[i].pt, kp2[i].pt, cv::Scalar(255, 0, 0));
                
            }
            
        }
        std::cout<<"\niteration"<<frameToRead-1<<"\n";//for testing purposes

        cv::Mat img;
        add(img1, mask, img);
        imshow("Frame", img);
        cv::waitKey(0);
        // Update the previous frame and points for next iteration    
        if(!goodNew.empty())
            kp1=goodNew; //transfer good points as to track them in new iteration
        img1=img2.clone();
        
        
        
    }
    cap.release();
    return 0;
}

void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    cv::parallel_for_(cv::Range(0, kp1.size()), //this one use cv::parallel_for_ to parallelize the calculateOpticalFlow method
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        cv::Mat H = cv::Mat::zeros(2, 2, CV_64F);   // hessian
        cv::Mat b = cv::Mat::zeros(2, 1, CV_64F);   // bias
        cv::Mat J = cv::Mat::zeros(2, 1, CV_64F);   // jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = cv::Mat::zeros(2, 2, CV_64F);
                b = cv::Mat::zeros(2, 1, CV_64F);
            } else {
                // only reset b
                b = cv::Mat::zeros(2, 1, CV_64F);
            }

            cost = 0;

            // compute cost and jacobian 
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {
                        double dI_dx = 0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                              GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y));
                        double dI_dy = 0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                              GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1));
                        // Create a cv::Mat for the Jacobian vector
                        J = (cv::Mat_<double>(2, 1) << -dI_dx, -dI_dy);
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        double dI_dx = 0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                          GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y));
                        double dI_dy = 0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                          GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1));
                        // Create a cv::Mat for the Jacobian vector
                        J = (cv::Mat_<double>(2, 1) << -dI_dx, -dI_dy);
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                         cv::Mat J_transpose;
                        cv::transpose(J, J_transpose);

                        // Compute J * J^T and add it to matrix H
                        cv::Mat J_JT = J * J_transpose;
                        H += J_JT;
                    }
                   
                }

            // compute update
            cv::Mat update;
            cv::solve(H, b, update, cv::DecompTypes::DECOMP_LU);
            

            if (std::isnan(update.at<double>(0))) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                std::cout << "update is nan" << "\n";
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update.at<double>(0);
            dy += update.at<double>(1);
            lastCost = cost;
            succ = true;
            double normValue = cv::norm(update, cv::NORM_L2);
            if (normValue < 1e-2) {
                // converge
                break;
            }
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    
    std::vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    

    // coarse-to-fine LK tracking in pyramids
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        

        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}
/*
int main(int, char**){
   
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
     

}
*/