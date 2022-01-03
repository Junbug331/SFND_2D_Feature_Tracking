# SFND 2D Feature Tracking

## Project Specification 
1. Data Buffer Optimization 
 * I implemented a ring buffer data structure in dataStructures.h. It is a template class which one can choose any data type or custom class or structure one sees fit. In order to implement the shifting feature while avoiding excessive copying of elements, I chose a linked list to make shifting process O(1) complexity since it can be easily done by rearragning data pointer(root node). More detailed explanation is provided in comments in dataStructures.h file. I also added custom Iterator class in order to make the code run without changing main code.

2. Keypoint Detection. 
 * I implemented detectors(HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT) in matching2D_student.cpp. In _void detKeypointsModern()_ function, I made it selectable by a string parameter, and used a switch statement to run selected detector function. I made enum and map dataStructures for detectors and descriptors for the faster comparison in switch statement. 

3. Keypoint Removal 
 * I used _contain()_ member function of _cv::Rect_ to check if a coordinate of the keypoint is within the rectangle. I made a temporary vector of keypoints and push all the keypoints that are in the rectangle and set this vector as our new keypoint vector.

4. Keypoint Descriptors 
 * I implemented descriptors(BRIEF, ORB, FREAK, AKAZE, and SIFT) in _void descKeypoints_ function and make them selectable by a string parameter. I also used a switch statement to make a selected descriptor to run. 

5. Descriptor Matching
 * I implemented FLANN matching and K-Nearest-Neighbor selection in _void matchDescriptors()_ function. Both methods are selectable via string parameters. 

6. Descriptor Distance Ratio
 * The descriptor distance ratio test in K-Nearest-Neghibor selection was implemented and the minimum ratio was set to 0.8. 

7. Performance Evaluation 1
Neighborhood size is the size of the scale space in which the keypoint was detected.
I chose the variance of the neighborhood size of all the keypoints found to be the measurement for distribution of their neighborhood size.
 * HARRIS
     - Number of keypoint : 248
     - Mean of the neighborhood size : 6
     - Distribution of neighborhood size(Variance of neighborhood size) : 0
 * FAST
     - Number of keypoint : 1491
     - Mean of the neighborhood size : 7
     - Distribution of neighborhood size(Variance of neighborhood size) : 0
 * BRISK
     - Number of keypoints : 2762
     - Mean of the neighborhood size : 22.0389
     - distribution of neighborhood size(Variance of neighborhood size) : 215.124 
     - Standard Deviation : 14.66 
 * ORB
     - Number of keypoints : 1003
     - Mean of the neighborhood size : 53.4901
     - distribution of neighborhood size(Variance of neighborhood size) : 490.492 
     - Standard Deviation : 22.14  
 * AKAZE
     - Number of keypoints : 1670
     - Mean of the neighborhood size : 7.885
     - distribution of neighborhood size(Variance of neighborhood size) : 12.9699 
     - Standard Deviation : 3.60
 * SIFT
     - Number of keypoints : 1386
     - Mean of the neighborhood size : 5.6251
     - distribution of neighborhood size(Variance of neighborhood size) : 44.638 
     - Standard Deviation : 6.68 

8. Performance Evaluation 2
    Included in data.csv file.

9. Performance Evaluation 3
 * Spreedsheet file is included in the project, named data.csv.
 * I made a seperate main file which runs simulation for all detector and descriptor combinations and writes a csv file. This file won't be included in CMakeListst.txt in submission. To log time it takes, I added extra parameters in keypoint detection functions and descriptor functions.
 * Since the object tracking will be perform in real driving situation, computation speed should match to that of real driving environment. Also, pictures taken while driving will vary depending speed of cars and road on which cars are driving. Speed would affect the scale of the image while road may affect orientation. Hence, fast algorithm which is invariant to scale and orientation should be selected. I discarded algorithms which aren't scale or rotation invariant. I also discared computationaly heavy algorithms. I selected -
    1. Detector: ORB, Descriptor: BRISK (combined time 48 ms)
    2. Detector: ORB, Descriptor: ORB (combined time 98 ms)
    3. Detector: ORB, Descriptor: FREAK (combined time 194 ms)

