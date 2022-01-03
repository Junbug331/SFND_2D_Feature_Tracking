#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
namespace DETECTORS
{
    enum type
    {
        HARRIS, FAST, BRISK, ORB, AKAZE, SIFT, SHITOMASI
    };

    static std::unordered_map<std::string, DETECTORS::type> detector_map{ {"HARRIS", HARRIS}, {"FAST", FAST}, {"BRISK", BRISK}, {"ORB", ORB}, {"AKAZE", AKAZE}, {"SIFT", SIFT}, {"SHITOMASI", SHITOMASI}};
}

namespace DESCRIPTORS
{
    // HOG: SIFT
    // Binary: BRIEF, BRISK, ORB, FREAK
    enum type
    {
        BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK
    };

    static std::unordered_map<std::string, DESCRIPTORS::type> descriptor_map { {"BRIEF", BRIEF}, {"ORB", ORB}, {"FREAK", FREAK}, {"AKAZE", AKAZE}, {"SIFT", SIFT}, {"BRISK", BRISK}};
}


void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log,bool bVis=false);
void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double &time_log, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType, double &time_log);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);



#endif /* matching2D_hpp */
