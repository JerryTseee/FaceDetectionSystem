//this header contains declarations for object detection-related functions and classes
//it includes functionality for detecting objects in images or video streams, like Haar cascades, HOG, and deep learning-based methods
#include "C:\opencv\build\include\opencv2\objdetect\objdetect.hpp"

//the highgui module provides functions for creating graphical user interfaces (GUIs) and handling input/output operations. It includes functions for displaying images, creating windows, capturing video from cameras, and handling keyboard/mouse events.
#include "C:\opencv\build\include\opencv2\highgui\highgui.hpp"

//the DNN module provides functionality for deep neural network (DNN) based operations
#include "C:\opencv\build\include\opencv2\dnn\dnn.hpp"


//image processing functions. It includes operations like resizing, filtering, edge detection, color conversion, and geometric transformations.
#include "C:\opencv\build\include\opencv2\imgproc\imgproc.hpp"
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;//use functions and classes from the OpenCV library


// Function Declaration to detect face in an image using OpenCV
void detectFace(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, dnn::Net& ageNet);

string cascadeName, nestedCascadeName;

/*
Mat& img: A reference to an OpenCV matrix (image).
CascadeClassifier& cascade: used for detecting the primary face features.
CascadeClassifier& nestedCascade: used for detecting nested features (like eyes within a face).
double scale: adjusts the size of the image during face detection.

cascadeName: A string variable that likely holds the path or name of the XML file containing the trained Haar cascade for face detection.
nestedCascadeName: Similarly, this string variable likely holds the path or name of the XML file for nested feature detection (e.g., eyes).
*/


int main(int argc, const char** argv)//argc and argv allow you to handle command-line arguments
{
    dnn::Net ageNet;//declare a variable named ageNet of type dnn::Net, it represents the neural network model for age estimation.
    //then, we load the pre-trained network from the Caffe framework.(Caffe is a deep learning framework that allow people to train neural networks for different tasks)
    ageNet = dnn::readNetFromCaffe("D:\\HKU_Resources\\C++tutorials\\C++training\\FaceDetection\\age_deploy.prototxt", "D:\\HKU_Resources\\C++tutorials\\C++training\\FaceDetection\\age_net.caffemodel");//"age_deploy.prototxt" file contains the architecture of the neural network. "age_net.caffemodel" file contains the learned weights and parameters of the neural network.
    
    // Class for video capturing from video files & provides C++ API for capturing video from cameras or reading video files.
    Mat frame, image;//declare two OpenCV matrices 


    // PreDefined trained XML classifiers with facial features
    CascadeClassifier faceCascade, eyeCascade;

    double scale = 3;//scale parameter controls the resizing of the input image during face detection. Adjusting this value can impact the detection performance.
    /*
    Increasing the scale factor makes the algorithm more sensitive to detecting smaller faces.
    help detect faces that are farther away or occupy a smaller portion of the image.

    Decreasing the scale factor makes the algorithm less sensitive to small faces.
    It focuses on larger faces and may miss smaller faces.
    */


    // Loads a classifiers from "opencv\sources\data\haarcascades" directory
    //loading eye cascade
    eyeCascade.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");
    //loading face cascade
    faceCascade.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalcatface.xml");




    // To capture Video from WebCam replace the path with 0, else edit the path
    //VideoCapture class in OpenCV is used for capturing video frames from video files or cameras.
    VideoCapture capture("C:\\Users\\admin\\Videos\\Jerry.mp4");


    if (capture.isOpened())//check whether the file is successfully opened
    {
        // Capturing frames from the video
        cout << "Face Detection is started...." << endl;
        while ( true )
        {
            capture >> frame;//frames are captured from the video
            if (frame.empty())//if the captured frame is empty(end of video), the loop breaks
                break;
            
            //A clone of the captured frame (frame1) is created for processing.
            Mat frame1 = frame.clone();

            // Calling the detect face function!!!
            detectFace(frame1, faceCascade, eyeCascade, scale, ageNet);//to detect faces in the cloned frame using the specified cascade classifiers (cascade and nestedCascade).
            

            char key = waitKey(1);//The program waits for a key press
            // Press q to exit from the window
            if (key == 'q' || key == 'Q')//If the pressed key is ．q・ or ．Q・, the loop breaks, allowing the user to exit the window.
                break;
        }
    }
    else
        cout << "Could not Open Video/Camera! ";
    return 0;
}


// Defination of Detect face function
void detectFace(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, dnn::Net& ageNet)
{
    vector<Rect> faces;
    Mat gray,smallImg;//Mat gray, smallImg;: These matrices are used for intermediate processing steps. gray is typically a grayscale version of the input image, and smallImg is a resized version of the image.

    // To convert the frames to gray color for better face detection
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);//enhance the contrast of the grayscale image

    double fx = 1.109 / scale;//fx>1: upsampling the image. fx<1: downsampling the image.


    // To resize the frames, smallImg matrix will contain the resized image.
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);//resizes the grayscale image (gray) to a smaller size.
    equalizeHist(smallImg, smallImg);


    // To detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    /*
    faces: A vector of rectangles (bounding boxes) that will store the detected faces.
    1.1: The scale factor. It determines how much the image size is reduced at each image scale. A value greater than 1 makes the algorithm more sensitive to smaller faces.
    2: The minimum number of neighboring rectangles (neighbors) required for a region to be considered a face. Increasing this value reduces false positives.
    0 | CASCADE_SCALE_IMAGE: Flags that control the detection process.
    Size(10,10): The minimum size of the detected faces. Faces smaller than this size will not be considered.
    */


    // To draw circles around the faces
    //The for loop iterates over the detected faces stored in the faces vector.
    //For each detected face, it performs the following steps :
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];//Rect r = faces[i]; extracts the bounding rectangle (face region) for the current detected face.
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(0, 255, 0);//this is for the circle color
        int radius;


        double aspect_ratio = (double)r.width / r.height;
        if (0.5 < aspect_ratio && aspect_ratio < 4)
        {
            //If the aspect ratio falls within the range of 0.75 to 1.3 (approximately square face), a circle is drawn around the face
            center.x = cvRound((r.x + r.width * 0.5) * scale);
            center.y = cvRound((r.y + r.height * 0.5) * scale);
            radius = cvRound((r.width + r.height) * 0.25 * scale);
            // To draw circle
            circle(img, center, radius, color, 3, 8, 0);
        }
        else
            rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),
                Point(cvRound((r.x + r.width - 1) * scale),
                    cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
        
        //If the nestedCascade is empty (not provided), the loop continues to the next detected face.
        if (nestedCascade.empty())
            continue;//next detected face
        smallImgROI = smallImg(r);

        //Age estimation part:
        Mat faceROI = img(r);//extract the face region from the image
        //Perform age estimation using the neural network model
        Mat blob = dnn::blobFromImage(faceROI, 1.0, Size(227, 227), Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
        ageNet.setInput(blob);
        //perform forward pass and get the predicted age
        Mat agePreds = ageNet.forward();
        int ageClass = max_element(agePreds.begin<float>(), agePreds.end<float>()) - agePreds.begin<float>();

        //since the model is not that accurate, I will manually add  back around 20
        ageClass += 20;

        //display the predicted age
        string ageLabel = "Age: " + to_string(ageClass);
        putText(img, ageLabel, Point(img.cols/2, img.rows/2), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
        //Point here is to draw the location of the display age label
    }


    // To display the detected face
    imshow("Face Detection", img);
}