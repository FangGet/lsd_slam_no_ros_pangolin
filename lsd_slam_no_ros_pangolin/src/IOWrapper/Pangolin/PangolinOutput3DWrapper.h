#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include <pangolin/pangolin.h>
#include <boost/thread.hpp>
#include <map>
#include <deque>
#include <Eigen/Core>

typedef Eigen::Matrix<float,3,1> Vec3f;
typedef Eigen::Matrix<unsigned int,int(3),int(1)> Vec3b;

namespace lsd_slam{
class Frame;
class KeyFrameGraph;
class KeyFrameDisplay;


struct GraphConnection
{
	Frame* from;
	Frame* to;
	float err;
};

class PangolinOutput3DWrapper:public Output3DWrapper{
public:

  
  PangolinOutput3DWrapper(int width,int height);
  virtual ~PangolinOutput3DWrapper();
  
  void run();
  void close();
  void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M,KeyFrameDisplay* kfd);
  
  virtual void publishKeyframeGraph(KeyFrameGraph* graph);

  // publishes a keyframe. if that frame already existis, it is overwritten, otherwise it is added.
  virtual void publishKeyframe(Frame* f);

  // published a tracked frame that did not become a keyframe (i.e. has no depth data)
  virtual void publishTrackedFrame(Frame* f);

  virtual void publishCamPose(Frame* frame);
 
  virtual void pushLiveFrame(cv::Mat& image);
  virtual void pushDepthImage(cv::Mat& image);

  virtual void publishDebugInfo(Eigen::Matrix<float, 20, 1> data);

  virtual void join();
  virtual void reset();
  
private:
  bool settings_followCamera;
  bool settings_showKFCameras;
  bool settings_showCurrentCamera;
  bool settings_showFullTrajectory;
  bool settings_showAllConstraints;
  
  bool settings_show3D;
  bool settings_showLiveDepth;
  bool settings_showLiveVideo;
  
  bool settings_resetButton;
  
  int width,height;
  bool needReset;
  void reset_internal();
  void drawConstraints();
  
  boost::thread runThread;
  bool running;
  int w,h;
  
  //images render;
  boost::mutex openImagesMutex;
  //need to add image definition
  char* depthImg;
  char* videoImg;
  
  bool videoImgChanged,depthImgChanged;
  
  //3D model rendering
  boost::mutex model3DMutex;
  KeyFrameDisplay* currentCam;
  std::vector<KeyFrameDisplay*> keyframes;
  std::vector<Vec3f> allFramePoses;
  std::vector<GraphConnection> connections;
  
  // timings
  struct timeval last_track;
  struct timeval last_map;
  
  std::deque<float> lastNTrackingMs;
  std::deque<float> lastNMappingMs;
  
};// PangolinOutput3DWrapper
  
}
