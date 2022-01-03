#include <numeric>
#include "matching2D.hpp"
#include <iostream>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<vector<cv::DMatch>> knn_matches;
        double minDescRatio = 0.8;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        for (auto& match : knn_matches)
        {
            if (match[0].distance < minDescRatio * match[1].distance)
                matches.push_back(match[0]);
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &time_log)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (DESCRIPTORS::descriptor_map.count(descriptorType) <= 0)
    {
        std::cout << "Valid descriptor types are [BRIEF, ORB, FREAK, AKAZE, SIFT]" << endl;
        std::cout << "Default descriptor type is BRISK" << endl;
    }
    switch(DESCRIPTORS::descriptor_map[descriptorType])
    {
        case DESCRIPTORS::BRISK:
        {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
            extractor = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }
        case DESCRIPTORS::BRIEF:
        {
             extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
             break;
        }
        case DESCRIPTORS::ORB:
        {
            int threshold = 30;
            extractor = cv:: ORB::create( 500, 1.2f, 8, 31, 0,2, cv::ORB::FAST_SCORE, 31, threshold);
            break;
        }
        case DESCRIPTORS::FREAK:
        {
            extractor = cv::xfeatures2d::FREAK::create();
            break;
        }
        case DESCRIPTORS::AKAZE:
        {
            extractor = cv::AKAZE::create();
            break;
        }
        case DESCRIPTORS::SIFT:
        {
            extractor = cv::SIFT::create();
            break;
        }
        default:
        {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
            extractor = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t * 1000 / 1.0;
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t * 1000 / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cout << "keypoints visualization" << endl;
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    // Detector Parameters;
    int blockSize = 2;
    int apertureSize = 3;
    int minResponse = 100; // R threshold
    double k = 0.04;

    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm;
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    double maxOverlap = 0.0;
    for (size_t j=0; j<dst_norm.rows; j++)
    {
        for (size_t i=0; i<dst_norm.cols; i++)
        {
            int response = static_cast<int>(dst_norm.at<float>(j,i));
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // NMS step
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                    keypoints.push_back(newKeyPoint);
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t * 1000 / 1.0;
    cout << "CornerHarris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    int threshold = 30;
    bool bNMS = true;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, cv::FastFeatureDetector::TYPE_9_16);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t * 1000 / 1.0;
    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{

    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t * 1000 / 1.0;
    cout << "BRISK(AGAST) detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    int FastThreshold = 30;
    cv::Ptr<cv::FeatureDetector> detector = cv:: ORB::create( 500, 1.2f, 8, 31, 0,2, cv::ORB::FAST_SCORE, 31, FastThreshold);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t*1000 / 1.0;
    cout << "ORB(FAST) detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t*1000 / 1.0;
    cout << "ORB(FAST) detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_log, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_log += t*1000 / 1.0;

    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

////  detectorType = [HARRIS, FAST, BRISK, ORB, AKAZE, SIFT]
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double &time_log, bool bVis)
{
    switch (DETECTORS::detector_map[detectorType])
    {
        case DETECTORS::HARRIS:
            detKeypointsHarris(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::FAST:
            detKeypointsFast(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::BRISK:
            detKeypointsBrisk(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::ORB:
            detKeypointsORB(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::AKAZE:
            detKeypointsAKAZE(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::SIFT:
            detKeypointsSIFT(keypoints, img, time_log, bVis);
            break;
        case DETECTORS::SHITOMASI:
            detKeypointsShiTomasi(keypoints, img, time_log, bVis);
            break;
        default:
            std::cout << "Available types are [HARRIS, FAST, BRISK, ORB, AKAZE, SIFT] " << std::endl;
            break;
    }
}
