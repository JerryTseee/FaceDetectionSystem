# FaceDetectionSystem  
implement a face detection system with Haar Cascade Algorithm, by using OpenCV in C++  

environment configuration: make sure to download OpenCV and Visual Studio in the computer.(eg. in here, my OpenCV file path is "C:\opencv")  
Below are the instructions in my case, please kindly modify them by yourselves.  
The Project is conducted in Visual Studio 2022  

Visual Studio 2022 Environment Configuration   
open Project Property Pages -> configuration properties -> debugging -> environment -> add the path: PATH=%PATH%;C:\opencv\build\x64\vc16\bin  
  
go to VC++ Directories -> include Directories -> add: C:\opencv\build\include;  
go to vc++ Directories -> Library Directories -> add: C:\opencv\build\x64\vc16\lib;  
go to C/C++ -> General -> Additional Include Directories -> add: C:\opencv\build\include;  
go to Linker -> General -> Additional Library Directories -> add: C:\opencv\build\x64\vc16\lib;  
go to Linker -> Input -> Additional Dependencies -> add: opencv_world490d.lib;  



Below is the testing result (age estimation is not accurate enough)     
<img width="344" alt="image" src="https://github.com/JerryTseee/FaceDetectionSystem/assets/126223772/d1170873-0ef3-4e35-a634-ddd167bd0bb0">  

Problem fixed: upload the wrong "age_deploy.prototxt" and "age_net.caffemodel"  :( but still not that good
<img width="347" alt="image" src="https://github.com/JerryTseee/FaceDetectionSystem/assets/126223772/96305725-05a6-4fc4-ae8f-39936aec24ce">

Since the model is not accurate, the age number will need to add back around 20  
<img width="352" alt="image" src="https://github.com/JerryTseee/FaceDetectionSystem/assets/126223772/d6818f32-3e33-4685-a9ac-03b320911530">
