#include "PangolinOutput3DWrapper.h"
#include "KeyFrameDisplay.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "DataStructures/Frame.h"

#include <opencv2/opencv.hpp>

namespace lsd_slam{
PangolinOutput3DWrapper::PangolinOutput3DWrapper(int width, int height)
{
  this->w=width;
  this->h=height;
  running=true;
  videoImg=new char[width*height*3];
  depthImg=new char[width*height*3];
  
  memset(videoImg, 128, sizeof(char)*w*h*3);
  memset(depthImg, 128., sizeof(char)*w*h*3);

  {
    currentCam = new KeyFrameDisplay();
  }
  
  needReset=false;
  runThread=boost::thread(&PangolinOutput3DWrapper::run,this);
}

PangolinOutput3DWrapper::~PangolinOutput3DWrapper()
{
  close();
  runThread.join();
}

void PangolinOutput3DWrapper::run()
{
  pangolin::CreateWindowAndBind("LSD_SLAM_VIEWER",2*w,2*h);
  const int UI_WIDTH=180;
  glEnable(GL_DEPTH_TEST);
  
  //3D visualization
  pangolin::OpenGlRenderState visualizer(
    pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
    pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY));
  pangolin::View& displayer=pangolin::CreateDisplay().SetBounds(0.0,1.0,pangolin::Attach::Pix(UI_WIDTH),1.0,-w/(float)h)
  .SetHandler(new pangolin::Handler3D(visualizer));
  
  //2images
  pangolin::View& d_video=pangolin::Display("imgVideo").SetAspect(w/(float)h);
  pangolin::View& d_depth=pangolin::Display("imgDepth").SetAspect(w/(float)h);
  
  pangolin::GlTexture texDepth(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  pangolin::GlTexture texVideo(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  
  pangolin::View& images_displayer=pangolin::CreateDisplay()
  .SetBounds(0.0,0.3,pangolin::Attach::Pix(UI_WIDTH),1.0)
  .SetLayout(pangolin::LayoutEqual)
  .AddDisplay(d_video)
  .AddDisplay(d_depth);
  
  //paramter reconfigure gui
  pangolin::CreatePanel("ui").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(UI_WIDTH));
  
  pangolin::Var<bool> settings_followCamera("ui.FollowCamera",false,true);//trouble here
  pangolin::Var<bool> settings_showKFCameras("ui.KFCam",true,true);
  pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam",true,true);
  //pangolin::Var<bool> settings_showTrajectory("ui.Trajectory",true,true);
  pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory",false,true);
  pangolin::Var<bool> settings_showAllConstraints("ui.AllConst",false,true);
  
  pangolin::Var<bool> settings_show3D("ui.show3D",true,true);
  pangolin::Var<bool> settings_showLiveDepth("ui.showDepth",true,true);
  pangolin::Var<bool> settings_showLiveVideo("ui.showVideo",true,true);

  
  pangolin::Var<double> settings_trackFps("ui.Track fps",0,0,0,false);
  pangolin::Var<double> settings_mapFps("ui.KF fps",0,0,0,false);
  
  pangolin::Var<bool> settings_resetButton("ui.Reset",false,false);
  
  bool camera_follow=true;
  
  while(!pangolin::ShouldQuit()&&running){
    //clear entire screen
    glClearColor(1.0f, 1.0f, 1.0f,0.6f); 
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    if(this->settings_show3D){
	//Active efficiently by object
      displayer.Activate(visualizer);
      boost::unique_lock<boost::mutex> lk3d(model3DMutex);
      int refreshed=0;
      for(KeyFrameDisplay* fh : keyframes){
	pangolin::OpenGlMatrix M;
	M.SetIdentity();
	getCurrentOpenGLCameraMatrix(M,fh);
	if(this->settings_followCamera&&camera_follow){
	  visualizer.Follow(M);
	}else if(this->settings_followCamera&&!camera_follow){
	  visualizer.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0));
	  visualizer.Follow(M);
	  camera_follow=true;
	}else if(!this->settings_followCamera&&camera_follow){
	  camera_follow=false;
	}

	float blue[3]={0,0,1};
	if(this->settings_showKFCameras)fh->drawCam(1,blue,0.2);
	refreshed+=(int)(fh->refreshPC(refreshed<8));
	fh->drawPC(true);
      }
      if(this->settings_showCurrentCamera)currentCam->drawCam(2,0,0.2);
      drawConstraints();
      lk3d.unlock();
    }
    
    {
      openImagesMutex.lock();
      if(videoImgChanged)
      {
	texVideo.Upload(videoImg,GL_RGB,GL_UNSIGNED_BYTE);
      }
      if(depthImgChanged){
	texDepth.Upload(depthImg,GL_RGB,GL_UNSIGNED_BYTE);
      }
      depthImgChanged=videoImgChanged=false;
      openImagesMutex.unlock();
    }
    
    {
      openImagesMutex.lock();
      float sd=0;
      for(float d:lastNMappingMs)sd+=d;
      settings_mapFps=lastNMappingMs.size()*1000.f/sd;
      openImagesMutex.unlock();
    }
    
    {
      model3DMutex.lock();
      float sd=0;
      for(float d:lastNTrackingMs)sd+=d;
      settings_trackFps=lastNTrackingMs.size()*1000.f/sd;
      model3DMutex.unlock();
    }
    
    if(this->settings_showLiveVideo){
      d_video.Activate();
      glColor4f(1.0f,1.0f,1.0f,1.0f);
      texVideo.RenderToViewportFlipY();
    }
    
    if(this->settings_showLiveDepth){
      d_depth.Activate();
      glColor4f(1.0f,1.0f,1.0f,1.0f);
      texDepth.RenderToViewportFlipY();
    }
    
    // update parameters
    this->settings_followCamera=settings_followCamera.Get();
    this->settings_showAllConstraints = settings_showAllConstraints.Get();
    this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
    this->settings_showKFCameras = settings_showKFCameras.Get();
    this->settings_showFullTrajectory = settings_showFullTrajectory.Get();
    
    this->settings_show3D=settings_show3D.Get();
    this->settings_showLiveDepth=settings_showLiveDepth.Get();
    this->settings_showLiveVideo=settings_showLiveVideo.Get();
    
    if(settings_resetButton.Get()){
      printf("Pangolin Viewer Reset ...\n");
      settings_resetButton.Reset();
    }
    
    pangolin::FinishFrame();
    if(needReset){
      reset_internal();
      camera_follow=true;
    }
  }
  printf("Pangolin is finished ... \n");
  
  exit(1);
}

void PangolinOutput3DWrapper::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M, KeyFrameDisplay* kfd)
{
  Sophus::Sim3f& cameraToWorld=kfd->camToWorld;
//   const Eigen::Matrix<float,int(3),int(3)>rotationToWorld=cameraToWorld.rotationMatrix().inverse();
//   Eigen::Matrix<float,int(3),int(1)> translationToWorld=cameraToWorld.translation();
//   translationToWorld=-rotationToWorld*translationToWorld;
  
//   Eigen::Matrix<float,4,4> camToWorld = cameraToWorld.matrix();
//   const Sophus::Matrix3f& rotationToWorld=camToWorld.block(0,0,3,3);
//   std::cout<<"0"<<std::endl;
//   Eigen::Matrix<float,int(3),int(1)> translationToWorld=(camToWorld.block(0,3,3,1));
//   std::cout<<"1"<<std::endl;
//   translationToWorld=-rotationToWorld*translationToWorld;
//   std::cout<<"2"<<std::endl;
//   M.m[0]=camToWorld(0,0);
//   M.m[1]=camToWorld(1,0);
//   M.m[2]=camToWorld(2,0);
//   M.m[3]=0.0;
//   
//   M.m[4]=camToWorld(0,1);
//   M.m[5]=camToWorld(1,1);
//   M.m[6]=camToWorld(2,1);
//   M.m[7]=0.0;
//   
//   M.m[8]=camToWorld(0,2);
//   M.m[9]=camToWorld(1,2);
//   M.m[10]=camToWorld(2,2);
//   M.m[11]=0.0;
//   std::cout<<"3"<<std::endl;
//   M.m[12]=translationToWorld(0);
//   M.m[13]=translationToWorld(1);
//   M.m[14]=translationToWorld(2);
//   M.m[15]=1.0;
  Eigen::Matrix<float,4,4> camToWorld=cameraToWorld.matrix();
  
  M.m[0]=camToWorld(0,0);
  M.m[1]=camToWorld(1,0);
  M.m[2]=camToWorld(2,0);
  M.m[3]=0.0;
  
  M.m[4]=camToWorld(0,1);
  M.m[5]=camToWorld(1,1);
  M.m[6]=camToWorld(2,1);
  M.m[7]=0.0;
  
  M.m[8]=camToWorld(0,2);
  M.m[9]=camToWorld(1,2);
  M.m[10]=camToWorld(2,2);
  M.m[11]=0.0;
  
  M.m[12]=camToWorld(0,3);
  M.m[13]=camToWorld(1,3);
  M.m[14]=camToWorld(2,3);
  M.m[15]=1.0;
  
}


void PangolinOutput3DWrapper::close()
{
  running=false;
}

void PangolinOutput3DWrapper::join()
{
  runThread.join();
  printf("JOINED Pangolin thread...\n");
}


void PangolinOutput3DWrapper::reset(){
  needReset=true;
}

void PangolinOutput3DWrapper::reset_internal()
{
  model3DMutex.lock();
  for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
  keyframes.clear();
  allFramePoses.clear();
  connections.clear();
  model3DMutex.unlock();
  
  openImagesMutex.lock();
  delete videoImg;videoImg=0;
  delete depthImg;depthImg=0;
  videoImgChanged= depthImgChanged=true;
  openImagesMutex.unlock();

  needReset = false;
}

void PangolinOutput3DWrapper::drawConstraints()
{
  if(this->settings_showAllConstraints){
    glLineWidth(1);
    
    glColor3f(1,0,1);
    glBegin(GL_LINES);
    for(unsigned int i=0;i<connections.size();i++){
     // if(connections[i].to->id()==0||connections[i].from->id()==0)continue;
      Sophus::Vector3f t=connections[i].from->getScaledCamToWorld().translation().cast<float>();
      glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
      t = connections[i].to->getScaledCamToWorld().translation().cast<float>();
      glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
    }
    glEnd();
  }
  
  if(this->settings_showFullTrajectory){
    float colorBlack[3]={0,1,0};
    glColor3f(colorBlack[0],colorBlack[1],colorBlack[2]);
    glLineWidth(3);
    
    glBegin(GL_LINE_STRIP);
    for(unsigned int i=0;i<allFramePoses.size();i++){
      glVertex3f((float)allFramePoses[i][0],
		 (float)allFramePoses[i][1],
		 (float)allFramePoses[i][2]);
    }
    glEnd();
  }
  
}

void PangolinOutput3DWrapper::publishKeyframeGraph(KeyFrameGraph* graph)
{
  if(!this->settings_show3D)return;
  model3DMutex.lock();
  connections.resize(graph->edgesAll.size());
  for(unsigned int i=0;i<graph->edgesAll.size();i++){
    connections[i].from=graph->edgesAll[i]->firstFrame;
    connections[i].to=graph->edgesAll[i]->secondFrame;
    Sophus::Vector7d err = graph->edgesAll[i]->edge->error();
    connections[i].err = sqrt(err.dot(err));
  }
  model3DMutex.unlock();
}

void PangolinOutput3DWrapper::publishKeyframe(Frame* frame)
{
  if(!this->settings_show3D)return;
  
  boost::unique_lock<boost::mutex> lk(model3DMutex);
  KeyFrameDisplay* kfd = new KeyFrameDisplay();
  kfd->setFromF(frame);
  keyframes.push_back(kfd);
}

void PangolinOutput3DWrapper::publishTrackedFrame(Frame* f)
{
//    if(!this->settings_show3D)return;
//    boost::unique_lock<boost::mutex> lk(model3DMutex);
//    KeyFrameDisplay* kfd = new KeyFrameDisplay();
//    kfd->setFromTrackedF(f);
//    keyframes.push_back(kfd);//to be checked
}


void PangolinOutput3DWrapper::publishCamPose(Frame* frame)
{
  boost::unique_lock<boost::mutex> lk(model3DMutex);
  struct timeval time_now;
  gettimeofday(&time_now, NULL);
  lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
  if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
  last_track = time_now;

  if(!this->settings_show3D) return;
  if(frame->hasIDepthBeenSet())
    currentCam->setFromF(frame);
  else
    currentCam->setFromTrackedF(frame);
  allFramePoses.push_back(frame->pose->getCamToWorld().translation().cast<float>()); 
}


void PangolinOutput3DWrapper::pushLiveFrame(cv::Mat& image){
//   if(!this->settings_showLiveVideo)return;
   boost::unique_lock<boost::mutex> lk(openImagesMutex);
   
   memcpy(videoImg,image.data,3*w*h);

   videoImgChanged=true;
}

void PangolinOutput3DWrapper::pushDepthImage(cv::Mat& image)
{
  boost::unique_lock<boost::mutex> lk(openImagesMutex);

  struct timeval time_now;
  gettimeofday(&time_now, NULL);
  lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
  if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
  last_map = time_now;

  memcpy(depthImg,image.data,3*w*h);
  depthImgChanged=true;
}

void PangolinOutput3DWrapper::publishDebugInfo(Eigen::Matrix< float, int(20), int(1) > data)
{
  //
}

}// lsd_slam
