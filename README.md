# FaceDetectionSystem
implement a face detection system by using OpenCV in C++

environment configuration: make sure to download OpenCV in the computer.(eg. here my OpenCV file path is "C:\opencv")
Below are the instructions in my case, please kindly modify them by yourselves.

open Project Property Pages -> configuration properties -> debugging -> environment -> add the path: PATH=%PATH%;C:\opencv\build\x64\vc16\bin
go to VC++ Directories -> include Directories -> add: C:\opencv\build\include;
go to vc++ Directories -> Library Directories -> add: C:\opencv\build\x64\vc16\lib;
go to C/C++ -> General -> Additional Include Directories -> add: C:\opencv\build\include;
go to Linker -> General -> Additional Library Directories -> add: C:\opencv\build\x64\vc16\lib;
go to Linker -> Input -> Additional Dependencies -> add: opencv_world490d.lib;
