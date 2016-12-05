/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LiveSLAMWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/Undistorter.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"

#include "IOWrapper/Pangolin/PangolinOutput3DWrapper.h"

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace lsd_slam;
int main( int argc, char** argv )
{
      if(argc!=2){
	  std::cerr<<std::endl<< 
	  "Usage: ./live_lsd_slam path_to_settings" 
	  << std::endl;
	  return -1;
      }

	std::string calibFile= std::string(argv[1]);
	Undistorter* undistorter = Undistorter::getUndistorterForFile(calibFile.c_str());

	if(undistorter == 0)
	{
		std::cerr<<"need camera calibration file..."<<std::endl;
		return -1;
	}

	int w = undistorter->getOutputWidth();
	int h = undistorter->getOutputHeight();

	int w_inp = undistorter->getInputWidth();
	int h_inp = undistorter->getInputHeight();

	float fx = undistorter->getK().at<double>(0, 0);
	float fy = undistorter->getK().at<double>(1, 1);
	float cx = undistorter->getK().at<double>(2, 0);
	float cy = undistorter->getK().at<double>(2, 1);
	Sophus::Matrix3f K;
	K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

	// make output wrapper. just set to zero if no output is required.
	Output3DWrapper* outputWrapper = new PangolinOutput3DWrapper(w,h);

	// make slam system
	SlamSystem* system = new SlamSystem(w, h, K, doSlam);
	system->setVisualization(outputWrapper);
	
	cv::VideoCapture capture(0);

	if(!capture.isOpened()){
	  std::cerr<<"no camera founded in this device ..."<<std::endl;
	  return -1;
	}
	
	cv::Mat frame;
	cv::Mat frame_original;
	int runningIDX=0;
	float fakeTimeStamp = 0;
	while(capture.read(frame_original)){
	  if(!frame_original.data){
	    std::cerr<<"no frame captured ..."<<std::endl;
	    continue;
	  }

	  if(frame_original.channels()==3){
	    cv::cvtColor(frame_original,frame_original,CV_BGR2RGB);
	    outputWrapper->pushLiveFrame(frame_original);
	    cv::cvtColor(frame_original,frame_original,CV_RGB2GRAY);
	  }

	  if(frame_original.rows != h_inp || frame_original.cols != w_inp)
	  {
	    printf("image has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
		    w,h,frame_original.cols, frame_original.rows);
	    continue;
	  }
	  assert(frame_original.type() == CV_8U);

	  undistorter->undistort(frame_original, frame);
	  
	  assert(frame.type() == CV_8U);

	  if(runningIDX == 0)
	    system->randomInit(frame.data, fakeTimeStamp, runningIDX);
	  else
	    system->trackFrame(frame.data, runningIDX ,false,fakeTimeStamp);
	  runningIDX++;
	  fakeTimeStamp+=0.03;

	  if(fullResetRequested)
	  {
	    printf("FULL RESET!\n");
	    delete system;
	    system = new SlamSystem(w, h, K, doSlam);
	    system->setVisualization(outputWrapper);
	    
	    fullResetRequested = false;
	    runningIDX = 0;
	  }
	  
      }


	system->finalize();

	delete system;
	delete undistorter;
	delete outputWrapper;
	return 0;
}
