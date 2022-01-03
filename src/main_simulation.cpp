#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <chrono>
#include <numeric>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

vector<pair<string, string>> make_combination(const vector<string> &detectors, const vector<string> &descriptors)
{
    vector<pair<string, string>> comb;
    for (const auto& det : detectors)
    {
        for (const auto& desc : descriptors)
        {
            if (desc == "AKAZE" && det != "AKAZE" || desc == "ORB" && det == "SIFT")
                continue;
            comb.push_back(make_pair(det, desc));
        }
    }
    return comb;
}

void print_combination(const vector<pair<string, string>> &comb)
{
    for (auto& elem : comb)
        cout << "["<< elem.first << ", " << elem.second << "]" << " ";
    cout << endl;
}

void run_simulation(vector<pair<string, string>> det_desc_pairs)
{
    string dataPath = "../";

    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)


    bool bVis = false;            // visualize results

    int vec_size = det_desc_pairs.size();
    vector<double> num_keypoints(vec_size, 0);
    vector<double> num_matches(vec_size, 0);
    vector<double> detection_time(vec_size, 0);
    vector<double> extration_time(vec_size, 0);

    for (int i=0; i<vec_size; i++)
    {
        // misc
        int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
        DataBuffer<DataFrame> dataBuffer(dataBufferSize);
        cout << "[Detector: " <<  det_desc_pairs[i].first << ", " << "Descriptor: " << det_desc_pairs[i].second << "] start!" << endl;
        for (size_t imgIndex = 0; imgIndex <= imgEndIndex; imgIndex++)
        {
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            DataFrame frame;
            frame.cameraImg = imgGray;
            dataBuffer.push_back(frame);

            vector<cv::KeyPoint> keypoints;
            string detectorType = det_desc_pairs[i].first;


            /************ feature detection ******************/
            detKeypointsModern(keypoints, imgGray, detectorType, detection_time[i], bVis);
           bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
                vector<cv::KeyPoint> newkeypoints;
                for (const auto& kp : keypoints)
                {
                    if (vehicleRect.contains(kp.pt))
                        newkeypoints.push_back(kp);
                }
                keypoints = std::move(newkeypoints);
            }

            // Log number of keypoints
            num_keypoints[i] += keypoints.size();
            (dataBuffer.end() -1)->keypoints = keypoints;


            /************ feature description ******************/
            cv::Mat descriptors;
            string descriptorType = det_desc_pairs[i].second;
            descKeypoints((dataBuffer.end()-1)->keypoints, (dataBuffer.end()-1)->cameraImg, descriptors, descriptorType, extration_time[i]);
            (dataBuffer.end()-1)->descriptors = descriptors;

            /************ Matching Keypoints ******************/
            if (dataBuffer.size() > 1)
            {
                vector<cv::DMatch> matches;
                string matcherType = "MAT_BF";
                string descriptorType = (det_desc_pairs[i].second.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY";
                string selectorType = "SEL_KNN";

                matchDescriptors((dataBuffer.end()-2)->keypoints, (dataBuffer.end()-1)->keypoints, (dataBuffer.end()-2)->descriptors, (dataBuffer.end()-1)->descriptors, matches, descriptorType, matcherType, selectorType);

                // log the number of matches
                num_matches[i] += matches.size();
                (dataBuffer.end() - 1)->kptMatches = matches;
            }

            // Neighborhood distribution
            if (imgIndex == imgEndIndex)
            {
                double sum_ = 0;
                for (auto &kp : keypoints)
                    sum_ += kp.size;
                double mean = sum_ / (double)keypoints.size();
                sum_ = 0.0;
                for (auto &kp: keypoints)
                    sum_ += pow(kp.size - mean, 2);
                double variance = sum_ / keypoints.size();

                cout << "Mean is " << mean << endl;
                cout << "Keypoint scale variance is " << variance << endl;
                cout << "Number of keypoint is " << num_keypoints[i] << endl;
                cout << "Number of matches is " << num_matches[i] << endl;
            }
        }
        cout << "[Detector: " <<  det_desc_pairs[i].first << ", " << "Descriptor: " << det_desc_pairs[i].second << "] end!" << endl;
    }

    // Make a name(det-desc) vector
    vector<string> pair_name;
    for (auto& elem : det_desc_pairs)
    {
        string name = "detector: " + elem.first + " descriptor: " + elem.second;
        pair_name.push_back(name);
    }

    // make a csv file
    ofstream fs("data.csv");
    fs << "Detector-Descriptor" << ",";
    fs << "Number of matches" << ",";
    fs << "Total detection time(ms)" << ",";
    fs << "Total extraction time(ms)" << "\n";

    for (int i=0; i<vec_size; i++)
    {
        fs << pair_name[i] << ",";
        fs << num_matches[i] << ",";
        fs << detection_time[i] << ",";
        fs << extration_time[i] << "\n";
    }

    fs.close();
}


int main()
{
    vector<string> detectors = {"HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT", "SHITOMASI"};
    vector<string> describers = {"BRIEF", "ORB", "FREAK", "AKAZE", "SIFT", "BRISK"};
    vector<pair<string, string>> det_desc = make_combination(detectors, describers);
    print_combination(det_desc);

    //det_desc = { make_pair("HARRIS", "BRIEF"), make_pair("FAST", "BRIEF"), make_pair("BRISK", "BRIEF"), make_pair("ORB", "BRIEF"), make_pair("AKAZE", "AKAZE"), make_pair("SIFT", "BRIEF")};

    cout << "Run Simulation" << endl;
    auto start = chrono::steady_clock::now();
    run_simulation(det_desc);
    auto end = chrono::steady_clock::now();
    cout << "Number of simulation : " << det_desc.size() << endl;
    cout << "End Simulation (took " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << "ms.)"<< endl;

    return 0;
}