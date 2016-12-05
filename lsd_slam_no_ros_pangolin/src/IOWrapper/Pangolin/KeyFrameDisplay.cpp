#define GL_GLEXT_PROTOTYPES 1
#include <stdio.h>
#include "KeyFrameDisplay.h"
#include "DataStructures/Frame.h"
#include<pangolin/pangolin.h>
#include <Eigen/Core>

typedef Eigen::Matrix<float,3,1> Vec3f;
typedef Eigen::Matrix<unsigned char,3,1> Vec3b;

namespace lsd_slam{
  
KeyFrameDisplay::KeyFrameDisplay()
{
  id=0;
  originalInput=0;
  vertexBufferIdValid=false;
  glBuffersValid=false;
  
  camToWorld=Sophus::Sim3f();
  width=height=0;
  
  my_scaledTH=my_absTH=0;
  
  totalPoints=displayedPoints=0;
  
  numRefreshedAlready=0;
  
  numGLBufferPoints=0;
}

KeyFrameDisplay::~KeyFrameDisplay()
{

  if(originalInput != 0)
    delete[] originalInput;
}

void KeyFrameDisplay::setFromF(Frame* kf)
{
  memcpy(camToWorld.data(),kf->getScaledCamToWorld().cast<float>().data(),7*sizeof(float));
  fx=kf->fx();
  fy=kf->fy();
  cx=kf->cx();
  cy=kf->cy();
  fxi = 1/fx;
  fyi = 1/fy;
  cxi = -cx / fx;
  cyi = -cy / fy;
  
  width=kf->width();
  height=kf->height();
  id=kf->id();
  time=kf->timestamp();
  
  if(originalInput!=0)
    delete[] originalInput;
  originalInput=0;
  originalInput = new InputPointDense[width*height];
  const float* idepth=kf->idepth(0);
  const float* idepthVal=kf->idepthVar(0);
  const float* color=kf->image(0);
  
  for(int idx=0;idx < width*height; idx++)
  {
    originalInput[idx].idepth = idepth[idx];
    originalInput[idx].idepth_val = idepthVal[idx];
    originalInput[idx].color[0] = color[idx];
    originalInput[idx].color[1] = color[idx];
    originalInput[idx].color[2] = color[idx];
    originalInput[idx].color[3] = color[idx];
  }
  glBuffersValid = false;
}

void KeyFrameDisplay::setFromTrackedF(Frame* fs)
{
  fx=fs->fx();
  fy=fs->fy();
  cx=fs->cx();
  cy=fs->cy();
  fxi = 1/fx;
  fyi = 1/fy;
  cxi = -cx / fx;
  cyi = -cy / fy;
  
  width=fs->width();
  height=fs->height();
  id=fs->id();
  time=fs->timestamp();
  memcpy(camToWorld.data(),fs->getScaledCamToWorld().cast<float>().data(),7*sizeof(float));
  
//   SE3 camToWorld = se3FromSim3(fs->getScaledCamToWorld());
//   
//   if(camToWorld.unit_quaternion().w()<0){
//     camToWorld.unit_quaternion().x()*=-1;
//     camToWorld.unit_quaternion().y()*=-1;
//     camToWorld.unit_quaternion().z()*=-1;
//     camToWorld.unit_quaternion().w()*=-1;
//   }

  if(originalInput!=0)
    delete[] originalInput;
  originalInput=0;

  glBuffersValid = false;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh)
{
  bool paramsStillGood = my_scaledTH == 1 &&my_absTH == 1 &&
		my_scale*1.2 > camToWorld.scale() &&
		my_scale < camToWorld.scale()*1.2 &&
		my_minNearSupport == 9 &&
		my_sparsifyFactor == 3;

  if(glBuffersValid && (paramsStillGood || numRefreshedAlready > 10)) return false;
  numRefreshedAlready++;
  
  glBuffersValid = true;

  // if there are no vertices, done!
  if(originalInput == 0)
    return false;

  // make data
  Vec3f* tmpVertexBuffer = new Vec3f[width*height];
  Vec3b* tmpColorBuffer = new Vec3b[width*height];

  my_scaledTH =1;
  my_absTH = 1;
  my_scale = camToWorld.scale();
  my_minNearSupport = 9;
  my_sparsifyFactor = 1;
  vertexBufferNumPoints = 0;

  int total = 0, displayed = 0;
  for(int y=1;y<height-1;y++)
    for(int x=1;x<width-1;x++)
    {
  if(originalInput[x+y*width].idepth <= 0) continue;
    total++;

  if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;

  float depth = 1 / originalInput[x+y*width].idepth;
  float depth4 = depth*depth; depth4*= depth4;

  if(originalInput[x+y*width].idepth_val * depth4 > my_scaledTH)
    continue;

  if(originalInput[x+y*width].idepth_val * depth4 * my_scale*my_scale > my_absTH)
    continue;

  if(my_minNearSupport > 1)
  {
    int nearSupport = 0;
    for(int dx=-1;dx<2;dx++)
      for(int dy=-1;dy<2;dy++)
      {
	int idx = x+dx+(y+dy)*width;
	if(originalInput[idx].idepth > 0)
	{
	  float diff = originalInput[idx].idepth - 1.0f / depth;
	  if(diff*diff < 2*originalInput[x+y*width].idepth_val)
	  nearSupport++;
	}
      }

      if(nearSupport < my_minNearSupport)
	continue;
  }

  tmpVertexBuffer[vertexBufferNumPoints][0] = (x*fxi + cxi) * depth;
  tmpVertexBuffer[vertexBufferNumPoints][1] = (y*fyi + cyi) * depth;
  tmpVertexBuffer[vertexBufferNumPoints][2] = depth;

  tmpColorBuffer[vertexBufferNumPoints][2] = originalInput[x+y*width].color[0];
  tmpColorBuffer[vertexBufferNumPoints][1] = originalInput[x+y*width].color[1];
  tmpColorBuffer[vertexBufferNumPoints][0] = originalInput[x+y*width].color[2];

  vertexBufferNumPoints++;
  displayed++;
  }
  totalPoints = total;
  displayedPoints = displayed;
   
  if(vertexBufferNumPoints==0)
  {
    delete[] tmpColorBuffer;
    delete[] tmpVertexBuffer;
    return true;
  }
	
  numGLBufferGoodPoints = vertexBufferNumPoints;
  if(numGLBufferGoodPoints > numGLBufferPoints)
  {
    numGLBufferPoints = vertexBufferNumPoints*1.3;
    vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
    colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
    
  }
  vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
  colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
  
  // create new ones, static
  vertexBufferIdValid = true;
  delete[] tmpColorBuffer;
  delete[] tmpVertexBuffer;
  
  return true;
}

void KeyFrameDisplay::drawPC(float pointSize, float alpha)
{
  if(!vertexBufferIdValid||numGLBufferGoodPoints==0)
  {
    return; 
  }

  GLfloat LightColor[] = {1, 1, 1, 1};
  if(alpha < 1)
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    LightColor[0] = LightColor[1] = 0;
    glEnable(GL_LIGHTING);
    glDisable(GL_LIGHT1);

    glLightfv (GL_LIGHT0, GL_AMBIENT, LightColor);
  }
  else
  {
    glDisable(GL_LIGHTING);
  }

  glPushMatrix();

  Sophus::Matrix4f m = camToWorld.matrix();
  glMultMatrixf((GLfloat*)m.data());

  glPointSize(pointSize);

  colorBuffer.Bind();
  glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  vertexBuffer.Bind();
  glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
  glDisableClientState(GL_VERTEX_ARRAY);
  vertexBuffer.Unbind();

  glDisableClientState(GL_COLOR_ARRAY);
  colorBuffer.Unbind();

  glPopMatrix();

  if(alpha < 1)
  {
    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    LightColor[2] = LightColor[1] = LightColor[0] = 1;
    glLightfv (GL_LIGHT0, GL_AMBIENT_AND_DIFFUSE, LightColor);
  }  
}

void KeyFrameDisplay::drawCam(float lineWidth, float* color,float sizeFactor)
{
  if(width == 0)
    return;
  
  float sz=sizeFactor;

  glPushMatrix();

  Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
  glMultMatrixf((GLfloat*)m.data());

  if(color == 0)
    glColor3f(1,0,0);
  else
    glColor3f(color[0],color[1],color[2]);


  glLineWidth(lineWidth);
  glBegin(GL_LINES);
/////////////////////////////////////
///lsd-like
/////////////////////////////////////
//   glVertex3f(0,0,0);
//   glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
//   
//   glVertex3f(0,0,0);
//   glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
//   
//   glVertex3f(0,0,0);
//   glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
//   
//   glVertex3f(0,0,0);
//   glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
//   
//   glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
//   glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
// 
//   glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
//   glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
// 
//   glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
//   glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
// 
//   glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
//   glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
  
/////////////////////////////////////
///orb-like
/////////////////////////////////////
  const float& w=sz;
  const float h=w*0.75;
  const float z=w*0.6;

  glVertex3f(0,0,0);
  glVertex3f(w,h,z);
  
  glVertex3f(0,0,0);
  glVertex3f(w,-h,z);
  
  glVertex3f(0,0,0);
  glVertex3f(-w,-h,z);
  
  glVertex3f(0,0,0);
  glVertex3f(-w,h,z);
  
  glVertex3f(w,h,z);
  glVertex3f(w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(-w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(w,h,z);

  glVertex3f(-w,-h,z);
  glVertex3f(w,-h,z);
  
  glEnd();
  glPopMatrix();
}




  
}//lsd_slam
