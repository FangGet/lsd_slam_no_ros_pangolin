#pragma once

#undef Success
#include <Eigen/Core>
#include <sophus/sophus.hpp>
#include<pangolin/pangolin.h>

#include "util/SophusUtil.h"

#include <sstream>
#include <fstream>

namespace lsd_slam{
class Frame;

struct InputPointDense{
  float idepth;
  float idepth_val;
  unsigned char color[4];
};

class KeyFrameDisplay{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeyFrameDisplay();
  ~KeyFrameDisplay();
  
  void setFromF(Frame* fs);
  void setFromTrackedF(Frame* fs);
  
  bool refreshPC(bool canRefresh);
  
  void drawCam(float lineWidth=1,float* color =0,float sizeFactor=1);
  
  void drawPC(float pointSize=1, float alpha=1);
  
  int id;
  
  double time;
  
  int totalPoints,displayedPoints;
  
  Sophus::Sim3f camToWorld;
  
  inline bool operator<(const KeyFrameDisplay& other) const{
    return (id<other.id);
  }
  
private:
  //camera parameter
  float fx,fy,cx,cy;
  float fxi,fyi,cxi,cyi;
  int width,height;
  
  float my_scaledTH,my_absTH,my_scale;
  int my_minNearSupport;
  int my_sparsifyFactor;
  
  //pointcloud data & respective buffer;
  InputPointDense* originalInput;
  
  //buffer & how many
  GLuint vertexBufferId;
  int vertexBufferNumPoints;
  
  bool vertexBufferIdValid;
  
  bool glBuffersValid;
   int numGLBufferPoints;
   int numGLBufferGoodPoints;
   pangolin::GlBuffer vertexBuffer;
   pangolin::GlBuffer colorBuffer;
   
   //to be set to pangolin
   int numRefreshedAlready;
   
   bool needRefresh;
};//end of KeyFrameDisplay

}//namespace lsd_slam