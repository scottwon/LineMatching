 #include "linedetector.h"
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include"linedetector.h"
#include <arlsmat.h>
#include <arlssym.h>
#include <iostream>
#include <fstream>
#include <map>
#include <cstdio>
#include <stdio.h>
#include <limits>
#include <memory>
#include <math.h>
#include <ctime>
#include<GL/glut.h>
using namespace std;
using namespace cv;
static const double pixelZ=438.4;//438.4
static const double trans=4.5;//4.5
static const double parallelBound=0.90;//0.90
static const double distBound=4.0;//4.0
static const double coplanarBound=0.15;//0.15
GLfloat rtri=0.0;
ofstream out;
ofstream sim;
string fLeft,fRight;
Point2f getIntersection(SEGMENT& seg1,SEGMENT& seg2){
    double x=(seg1.x2-seg1.x1)*(seg2.x1*seg2.y2-seg2.x2*seg2.y1)-(seg2.x2-seg2.x1)*(seg1.x1*seg1.y2-seg1.y1*seg1.x2);
    double y=(seg1.y1-seg1.y2)*(seg2.x1*seg2.y2-seg2.y1*seg2.x2)-(seg2.y1-seg2.y2)*(seg1.x1*seg1.y2-seg1.y1*seg1.x2);
    double r=(seg2.x2-seg2.x1)*(seg1.y1-seg1.y2)-(seg2.y1-seg2.y2)*(seg1.x2-seg1.x1);
    x=x/r;
    y=(-1)*y/r;
    return Point2f(x,y);
}

bool checkIntersection(SEGMENT& seg1,SEGMENT& seg2){
    double x=(seg1.x2-seg1.x1)*(seg2.x1*seg2.y2-seg2.x2*seg2.y1)-(seg2.x2-seg2.x1)*(seg1.x1*seg1.y2-seg1.y1*seg1.x2);
    double r=(seg2.x2-seg2.x1)*(seg1.y1-seg1.y2)-(seg2.y1-seg2.y2)*(seg1.x2-seg1.x1);
    x=x/r;
    double X1min=min(seg1.x1,seg1.x2);
    double X1max=max(seg1.x1,seg1.x2);
    double X2min=min(seg2.x1,seg2.x2);
    double X2max=max(seg2.x1,seg2.x2);
    double delta1=(X1max-X1min)/100;
    double delta2=(X2max-X2min)/100;
    if(x>X1min+delta1 && x<X1max-delta1 && x>X2min+delta2 && x<X2max-delta2){
        return true;
    }
    else{
        return false;
    }
}

struct homogeneousPoint{
    double x,y,z,w;
    homogeneousPoint(){
        x=0;y=0;z=0;w=1;
    }
    homogeneousPoint(double px,double py,double pz=pixelZ){
        x=px;
        y=py;
        z=pz;
        w=1;
    }
};

struct normalVector{
    double nx,ny,nz;
    normalVector(){
        nx=0;ny=0;nz=0;
    }
    normalVector(double px,double py,double pz){
        nx=px;ny=py;nz=pz;
    }
    bool normalize(){
        double norm=sqrt(nx*nx+ny*ny+nz*nz);
        if(norm==0)return false;
        nx/=norm;
        ny/=norm;
        nz/=norm;
        return false;
    }
};

normalVector crossProduct(const homogeneousPoint &p1,const homogeneousPoint &p2){
    normalVector v(p1.y/p1.w*p2.z/p2.w-p2.y/p2.w*p1.z/p1.w,p1.z/p1.w*p2.x/p2.w-p2.z/p2.w*p1.x/p1.w,p1.x/p1.w*p2.y/p2.w-p1.y/p1.w*p2.x/p2.w);
    v.normalize();
    return v;
}

normalVector crossProduct(const normalVector &v1,const normalVector &v2){
    normalVector v(v1.ny*v2.nz-v1.nz*v2.ny,v1.nz*v2.nx-v1.nx*v2.nz,v1.nx*v2.ny-v2.nx*v1.ny);
    v.normalize();
    return v;
}

struct motionTranslation{
    double tx,ty,tz;
    motionTranslation(){
        tx=0;ty=0;tz=0;
    }
    motionTranslation(double px,double py,double pz){
        tx=px;ty=py;tz=pz;
    }
};

homogeneousPoint translation(const homogeneousPoint& p1,const motionTranslation& m){
    return homogeneousPoint(p1.x/p1.w+m.tx,p1.y/p1.w+m.ty,p1.z/p1.w+m.tz);
}

normalVector getNormalVector(const SEGMENT& seg,const Mat& img){
    return crossProduct(homogeneousPoint(seg.x1-(img.cols-1)/2,(img.rows-1)/2-seg.y1),homogeneousPoint(seg.x2-(img.cols-1)/2,(img.rows-1)/2-seg.y2));
}

normalVector getLineDirectionVector(const SEGMENT& seg1,const SEGMENT& seg2,const Mat& img){
    return crossProduct(getNormalVector(seg1,img),getNormalVector(seg2,img));
}

struct line3D{
    Vec3f direction;
    Vec3f leftEndPoint1,leftEndPoint2,rightEndPoint1,rightEndPoint2,leftMidPoint,rightMidPoint;
    Vec3f color;
    int ID;
};

struct lineRelations{
    double angle;
    double dist;
};

double getAngle(const line3D& l1, const line3D& l2){
    return l1.direction[0]*l2.direction[0]+l1.direction[1]*l2.direction[1]+l1.direction[2]*l2.direction[2];
}

double getDist(const line3D& l1,const line3D& l2){
    normalVector n1(l1.direction[0],l1.direction[1],l1.direction[2]);
    normalVector n2(l2.direction[0],l2.direction[1],l2.direction[2]);
    normalVector n=crossProduct(n1,n2);
    normalVector v(l2.leftEndPoint1[0]-l1.leftEndPoint1[0],l2.leftEndPoint1[1]-l1.leftEndPoint1[1],l2.leftEndPoint1[2]-l1.leftEndPoint1[2]);
    return abs(v.nx*n.nx+v.ny*n.ny+v.nz*n.nz);
}

Vec3f getNormalVector(const line3D& l1,const line3D& l2){
    Vec3f tmp;
    double len;
    tmp[0]=l1.direction[1]*l2.direction[2]-l1.direction[2]*l2.direction[1];
    tmp[1]=l1.direction[2]*l2.direction[0]-l1.direction[0]*l2.direction[2];
    tmp[2]=l1.direction[0]*l2.direction[1]-l1.direction[1]*l2.direction[0];
    len=sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
    tmp[0]=tmp[0]/len;
    tmp[1]=tmp[1]/len;
    tmp[2]=tmp[2]/len;
    return tmp;
}

double getAngle(const line3D& l,const Vec3f& nv){
    double tmp=l.direction[0]*nv[0]+l.direction[1]*nv[1]+l.direction[2]*nv[2];
    double len=sqrt(l.direction[0]*l.direction[0]+l.direction[1]*l.direction[1]+l.direction[2]*l.direction[2]);
    return tmp/len;
}

void cvFitPlane(const Mat& points, float* plane){  
    int nrows = points.rows;  
    int ncols = points.cols;  
    Mat centroid;
    centroid.create(1, ncols, CV_32FC1);  
    for(int i=0;i<ncols;i++){
        centroid.at<float>(i)=0;
    }
    for (int c = 0; c<ncols; c++){
        for (int r = 0; r < nrows; r++)  
        {  
            centroid.at<float>(c) += points.at<float>(r, c);  
        }  
        centroid.at<float>(c) /= nrows;  
    }  
    Mat points2;
    points2.create(nrows, ncols, CV_32FC1);  
    for (int r = 0; r<nrows; r++)  
        for (int c = 0; c<ncols; c++)  
            points2.at<float>(r,c) = points.at<float>(r, c) - centroid.at<float>(c);  
    Mat A,W,U,V;
    A.create(ncols, ncols, CV_32FC1);  
    W.create(ncols, ncols, CV_32FC1);  
    V.create(ncols, ncols, CV_32FC1); 
    gemm(points2, points, 1, NULL, 0, A, GEMM_1_T);
    SVD::compute(A, W, U, V);  
    plane[ncols] = 0;  
    for (int c = 0; c<ncols; c++){  
        plane[c] = V.at<float>(ncols-1,c);  
        plane[ncols] += plane[c] * centroid.at<float>(c);  
    }
    points2.release();
    A.release();
    W.release();
    U.release();
    V.release();
}  

void fitPlane(vector<line3D>& coplanarLines,float* plane){
    Mat pointsMat;
    pointsMat.create(coplanarLines.size()*3,3,CV_32FC1);
    for(int i=0;i<coplanarLines.size();i++){
        pointsMat.at<float>((3*i),0)=coplanarLines[i].leftEndPoint1[0];
        pointsMat.at<float>((3*i),1)=coplanarLines[i].leftEndPoint1[1];
        pointsMat.at<float>((3*i),2)=coplanarLines[i].leftEndPoint1[2];
        pointsMat.at<float>((3*i+1),0)=coplanarLines[i].leftEndPoint2[0];
        pointsMat.at<float>((3*i+1),1)=coplanarLines[i].leftEndPoint2[1];
        pointsMat.at<float>((3*i+1),2)=coplanarLines[i].leftEndPoint2[2];
        pointsMat.at<float>((3*i+2),0)=coplanarLines[i].leftMidPoint[0];
        pointsMat.at<float>((3*i+2),1)=coplanarLines[i].leftMidPoint[1];
        pointsMat.at<float>((3*i+2),2)=coplanarLines[i].leftMidPoint[2];
    }
    cvFitPlane(pointsMat,plane);
}

bool fitLine(const line3D& l,float* plane,double bound=distBound){
    if(abs(l.leftEndPoint1[0]*plane[0]+l.leftEndPoint1[1]*plane[1]*l.leftEndPoint1[2]*plane[2]-plane[3])>distBound)return false;
    if(abs(l.leftEndPoint2[0]*plane[0]+l.leftEndPoint2[1]*plane[1]+l.leftEndPoint2[2]*plane[2]-plane[3])>distBound)return false;
    return true;
}

typedef vector<line3D> PlaneRecon;
typedef vector<PlaneRecon> structure3D;

double det(Mat m){
    double r=0.0;
    r+=m.at<float>(0,0)*m.at<float>(1,1)*m.at<float>(2,2);
    r+=m.at<float>(0,1)*m.at<float>(1,2)*m.at<float>(2,0);
    r+=m.at<float>(0,2)*m.at<float>(1,0)*m.at<float>(2,1);
    r-=m.at<float>(0,2)*m.at<float>(1,1)*m.at<float>(2,0);
    r-=m.at<float>(0,1)*m.at<float>(1,0)*m.at<float>(2,2);
    r-=m.at<float>(0,0)*m.at<float>(1,2)*m.at<float>(2,1);
    return r;
}

void getCoef(Mat a,Mat b,double& c0,double& c1,double& c2,double& c3){
    c0=det(a);
    c3=det(b)*(-1);
    Mat t1,t2,t3,t4,t5,t6;
    t1.create(3,3,CV_32FC1);
    t2.create(3,3,CV_32FC1);
    t3.create(3,3,CV_32FC1);
    t4.create(3,3,CV_32FC1);
    t5.create(3,3,CV_32FC1);
    t6.create(3,3,CV_32FC1);
    t1.at<float>(0,0)=b.at<float>(0,0);
    t1.at<float>(1,0)=b.at<float>(1,0);
    t1.at<float>(2,0)=b.at<float>(2,0);
    t1.at<float>(0,1)=a.at<float>(0,1);
    t1.at<float>(1,1)=a.at<float>(1,1);
    t1.at<float>(2,1)=a.at<float>(2,1);
    t1.at<float>(0,2)=a.at<float>(0,2);
    t1.at<float>(1,2)=a.at<float>(1,2);
    t1.at<float>(2,2)=a.at<float>(2,2);
    t2.at<float>(0,0)=a.at<float>(0,0);
    t2.at<float>(1,0)=a.at<float>(1,0);
    t2.at<float>(2,0)=a.at<float>(2,0);
    t2.at<float>(0,1)=b.at<float>(0,1);
    t2.at<float>(1,1)=b.at<float>(1,1);
    t2.at<float>(2,1)=b.at<float>(2,1);
    t2.at<float>(0,2)=a.at<float>(0,2);
    t2.at<float>(1,2)=a.at<float>(1,2);
    t2.at<float>(2,2)=a.at<float>(2,2);
    t3.at<float>(0,0)=a.at<float>(0,0);
    t3.at<float>(1,0)=a.at<float>(1,0);
    t3.at<float>(2,0)=a.at<float>(2,0);
    t3.at<float>(0,1)=a.at<float>(0,1);
    t3.at<float>(1,1)=a.at<float>(1,1);
    t3.at<float>(2,1)=a.at<float>(2,1);
    t3.at<float>(0,2)=b.at<float>(0,2);
    t3.at<float>(1,2)=b.at<float>(1,2);
    t3.at<float>(2,2)=b.at<float>(2,2);
    t4.at<float>(0,0)=a.at<float>(0,0);
    t4.at<float>(1,0)=a.at<float>(1,0);
    t4.at<float>(2,0)=a.at<float>(2,0);
    t4.at<float>(0,1)=b.at<float>(0,1);
    t4.at<float>(1,1)=b.at<float>(1,1);
    t4.at<float>(2,1)=b.at<float>(2,1);
    t4.at<float>(0,2)=b.at<float>(0,2);
    t4.at<float>(1,2)=b.at<float>(1,2);
    t4.at<float>(2,2)=b.at<float>(2,2);
    t5.at<float>(0,0)=b.at<float>(0,0);
    t5.at<float>(1,0)=b.at<float>(1,0);
    t5.at<float>(2,0)=b.at<float>(2,0);
    t5.at<float>(0,1)=a.at<float>(0,1);
    t5.at<float>(1,1)=a.at<float>(1,1);
    t5.at<float>(2,1)=a.at<float>(2,1);
    t5.at<float>(0,2)=b.at<float>(0,2);
    t5.at<float>(1,2)=b.at<float>(1,2);
    t5.at<float>(2,2)=b.at<float>(2,2);
    t6.at<float>(0,0)=b.at<float>(0,0);
    t6.at<float>(1,0)=b.at<float>(1,0);
    t6.at<float>(2,0)=b.at<float>(2,0);
    t6.at<float>(0,1)=b.at<float>(0,1);
    t6.at<float>(1,1)=b.at<float>(1,1);
    t6.at<float>(2,1)=b.at<float>(2,1);
    t6.at<float>(0,2)=a.at<float>(0,2);
    t6.at<float>(1,2)=a.at<float>(1,2);
    t6.at<float>(2,2)=a.at<float>(2,2);
    c1=-det(t1)-det(t2)-det(t3);
    c2=det(t4)+det(t5)+det(t6);
}

Mat subtract(Mat m1,Mat m2){
    Mat m;
    m.create(3,3,CV_32FC1);
    m.at<float>(0,0)=m1.at<float>(0,0)-m2.at<float>(0,0);
    m.at<float>(0,1)=m1.at<float>(0,1)-m2.at<float>(0,1);
    m.at<float>(0,2)=m1.at<float>(0,2)-m2.at<float>(0,2);
    m.at<float>(1,0)=m1.at<float>(1,0)-m2.at<float>(1,0);
    m.at<float>(1,1)=m1.at<float>(1,1)-m2.at<float>(1,1);
    m.at<float>(1,2)=m1.at<float>(1,2)-m2.at<float>(1,2);
    m.at<float>(2,0)=m1.at<float>(2,0)-m2.at<float>(2,0);
    m.at<float>(2,1)=m1.at<float>(2,1)-m2.at<float>(2,1);
    m.at<float>(2,2)=m1.at<float>(2,2)-m2.at<float>(2,2);
    return m;
}

Mat multiply(Mat m1,Mat m2){
    Mat m;
    m.create(3,3,CV_32FC1);
    m.at<float>(0,0)=m1.at<float>(0,0)*m2.at<float>(0,0)+m1.at<float>(0,1)*m2.at<float>(1,0)+m1.at<float>(0,2)*m2.at<float>(2,0);
    m.at<float>(0,1)=m1.at<float>(0,0)*m2.at<float>(0,1)+m1.at<float>(0,1)*m2.at<float>(1,1)+m1.at<float>(0,2)*m2.at<float>(2,1);
    m.at<float>(0,2)=m1.at<float>(0,0)*m2.at<float>(0,2)+m1.at<float>(0,1)*m2.at<float>(1,2)+m1.at<float>(0,2)*m2.at<float>(2,2);
    m.at<float>(1,0)=m1.at<float>(1,0)*m2.at<float>(0,0)+m1.at<float>(1,1)*m2.at<float>(1,0)+m1.at<float>(1,2)*m2.at<float>(2,0);
    m.at<float>(1,1)=m1.at<float>(1,0)*m2.at<float>(0,1)+m1.at<float>(1,1)*m2.at<float>(1,1)+m1.at<float>(1,2)*m2.at<float>(2,1);
    m.at<float>(1,2)=m1.at<float>(1,0)*m2.at<float>(0,2)+m1.at<float>(1,1)*m2.at<float>(1,2)+m1.at<float>(1,2)*m2.at<float>(2,2);
    m.at<float>(2,0)=m1.at<float>(2,0)*m2.at<float>(0,0)+m1.at<float>(2,1)*m2.at<float>(1,0)+m1.at<float>(2,2)*m2.at<float>(2,0);
    m.at<float>(2,1)=m1.at<float>(2,0)*m2.at<float>(0,1)+m1.at<float>(2,1)*m2.at<float>(1,1)+m1.at<float>(2,2)*m2.at<float>(2,1);
    m.at<float>(2,2)=m1.at<float>(2,0)*m2.at<float>(0,2)+m1.at<float>(2,1)*m2.at<float>(1,2)+m1.at<float>(2,2)*m2.at<float>(2,2);
    return m;
}

Mat reverse(Mat m1){
    Mat m;
    m.create(3,3,CV_32FC1);
    double c=det(m1);
    m.at<float>(0,0)=m1.at<float>(1,1)*m1.at<float>(2,2)-m1.at<float>(1,2)*m1.at<float>(2,1);
    m.at<float>(0,1)=m1.at<float>(2,1)*m1.at<float>(0,2)-m1.at<float>(0,1)*m1.at<float>(2,2);
    m.at<float>(0,2)=m1.at<float>(0,1)*m1.at<float>(1,2)-m1.at<float>(1,1)*m1.at<float>(0,2);
    m.at<float>(1,0)=m1.at<float>(2,0)*m1.at<float>(1,2)-m1.at<float>(1,0)*m1.at<float>(2,2);
    m.at<float>(1,1)=m1.at<float>(0,0)*m1.at<float>(2,2)-m1.at<float>(2,0)*m1.at<float>(0,2);
    m.at<float>(1,2)=m1.at<float>(1,0)*m1.at<float>(0,2)-m1.at<float>(0,0)*m1.at<float>(1,2);
    m.at<float>(2,0)=m1.at<float>(1,0)*m1.at<float>(2,1)-m1.at<float>(1,1)*m1.at<float>(2,0);
    m.at<float>(2,1)=m1.at<float>(2,0)*m1.at<float>(0,1)-m1.at<float>(2,1)*m1.at<float>(0,0);
    m.at<float>(2,2)=m1.at<float>(0,0)*m1.at<float>(1,1)-m1.at<float>(0,1)*m1.at<float>(1,0);
    m.at<float>(0,0)/=c;
    m.at<float>(0,1)/=c;
    m.at<float>(0,2)/=c;
    m.at<float>(1,0)/=c;
    m.at<float>(1,1)/=c;
    m.at<float>(1,2)/=c;
    m.at<float>(2,0)/=c;
    m.at<float>(2,1)/=c;
    m.at<float>(2,2)/=c;
    return m;
}

double LieBracket(Mat Hi,Mat Hj,Mat Hk,Mat Hl,int n,int m){
    Mat m1=multiply(reverse(Hj),Hi);
    Mat m2=multiply(reverse(Hl),Hk);
    Mat s1=m1;
    Mat s2=m2;
    while(n>1){
        s1=multiply(s1,m1);
        --n;
    }
    while(m>1){
        s2=multiply(s2,m2);
        --m;
    }
    return det(subtract(multiply(s1,s2),multiply(s2,s1)));
}

double discriminant(double c0,double c1,double c2,double c3){
    double r;
    r=18*c0*c1*c2*c3-4*c2*c2*c2*c0+c2*c2*c1*c1-4*c3*c1*c1*c1-27*c3*c3*c0*c0;
    return r;
}

class stereo{
    public:
        static double focalLength;
        static Vec3f translation;
        Mat imgLeft,imgRight,imgLeftGray,imgRightGray;
        int imageWidth,imageHeight;
        LineDetector ld;
        vector<SEGMENT>linesLeft,linesRight;
        vector<unsigned int> matchResult;
        vector<segDesc> segDescLeft,segDescRight;
        vector<line3D>lineRecon;
        vector<normalVector>leftPlanes,rightPlanes;
        vector<double>colorR,colorG,colorB;
        vector<vector<lineRelations> >lineRelMat;
        vector<vector<Point2f> >featuresLeft,featuresRight;
        vector<vector<Mat> >homographies;
        structure3D planeStructure;
        vector<vector<float> >PlaneFitEqu;
        vector<Mat>PlaneHomographies;
        vector<Mat>constraintH;
//        vector<vector<Mat> >transImg;
    public:
        stereo(){
        }
        void set(const string& filenameLeft,const string& filenameRight){
            imgLeft=imread(filenameLeft);
            imgRight=imread(filenameRight);
            cvtColor(imgLeft,imgLeftGray,COLOR_RGB2GRAY);
            cvtColor(imgRight,imgRightGray,COLOR_RGB2GRAY);
            imageWidth=imgLeft.cols;
            imageHeight=imgLeft.rows;
            linesLeft.clear();
            linesRight.clear();
            ld.lineDetection(imgLeftGray,linesLeft);
            ld.lineDetection(imgRightGray,linesRight);
            descriptor desc(9,7);
            desc.getGradientMap(filenameLeft);
            segDescLeft.clear();
            segDescRight.clear();
	        segDescLeft.resize(linesLeft.size());
  	        segDescRight.resize(linesRight.size());
	        for(int i=0;i<linesLeft.size();i++){
	        	segDescLeft[i].segment.x1=linesLeft.at(i).x1;
  		        segDescLeft[i].segment.x2=linesLeft.at(i).x2;
	  	        segDescLeft[i].segment.y1=linesLeft.at(i).y1;
		        segDescLeft[i].segment.y2=linesLeft.at(i).y2;
  		        segDescLeft[i].segment.angle=linesLeft.at(i).angle;
  		        segDescLeft[i].segment.label=i;
  		        segDescLeft[i].lineLength=sqrt((linesLeft.at(i).x2-linesLeft.at(i).x1)*(linesLeft.at(i).x2-linesLeft.at(i).x1)+(linesLeft.at(i).y2-linesLeft.at(i).y1)*(linesLeft.at(i).y2-linesLeft.at(i).y1));
  		        segDescLeft[i].desVec=desc.calcLBD(linesLeft.at(i));
  	        }
  	        desc.getGradientMap(filenameRight);
  	        for(int i=0;i<linesRight.size();i++){
  		        segDescRight[i].segment.x1=linesRight.at(i).x1;
  		        segDescRight[i].segment.x2=linesRight.at(i).x2;
  		        segDescRight[i].segment.y1=linesRight.at(i).y1;
  		        segDescRight[i].segment.y2=linesRight.at(i).y2;
  		        segDescRight[i].segment.angle=linesRight.at(i).angle;
  		        segDescRight[i].segment.label=i;
  		        segDescRight[i].lineLength=sqrt((linesRight.at(i).x2-linesRight.at(i).x1)*(linesRight.at(i).x2-linesRight.at(i).x1)+(linesRight.at(i).y2-linesRight.at(i).y1)*(linesRight.at(i).y2-linesRight.at(i).y1));
  		        segDescRight[i].desVec=desc.calcLBD(linesRight.at(i));
  	        }
	        BuildMat(segDescLeft,segDescRight);
            matchResult.clear();
	        MatchingResult(segDescLeft,segDescRight,matchResult);
            leftPlanes.clear();
            rightPlanes.clear();
            lineRecon.clear();
/*            srand((int)time(0));
            colorR.clear();
            colorG.clear();
            colorB.clear();
            colorR.resize(matchResult.size()/2);
            colorG.resize(matchResult.size()/2);
            colorB.resize(matchResult.size()/2);
            for(int i=0;i<matchResult.size()/2;i++){
                colorR[i]=((double)rand())/RAND_MAX;
                colorG[i]=((double)rand())/RAND_MAX;
                colorB[i]=((double)rand())/RAND_MAX;
            }
            Mat img=imread(filenameLeft);
            Mat img2=imread(filenameRight);
            string iid;
            for(unsigned int i=0;i<matchResult.size()/2;i++){
                stringstream sss;
                sss<<i+1;
                sss>>iid;
                line(img,Point(segDescLeft[matchResult[2*i]].segment.x1,segDescLeft[matchResult[2*i]].segment.y1),Point(segDescLeft[matchResult[2*i]].segment.x2,segDescLeft[matchResult[2*i]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
                //putText(img,iid,Point((segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2,(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
                line(img2,Point(segDescRight[matchResult[2*i+1]].segment.x1,segDescRight[matchResult[2*i+1]].segment.y1),Point(segDescRight[matchResult[2*i+1]].segment.x2,segDescRight[matchResult[2*i+1]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
                //putText(img2,iid,Point((segDescRight[matchResult[2*i+1]].segment.x1+segDescRight[matchResult[2*i+1]].segment.x2)/2,(segDescRight[matchResult[2*i+1]].segment.y1+segDescRight[matchResult[2*i+1]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
            }
            imshow("Left View",img);
            imshow("Right View",img2);
            imwrite("outleft.png",img);
            imwrite("outright.png",img2);
            waitKey(0);*/
            for(int i=0;i<matchResult.size()/2;i++){
                normalVector n1=getNormalVector(segDescLeft[matchResult[2*i]].segment,imgLeft);
                normalVector n2=getNormalVector(segDescRight[matchResult[2*i+1]].segment,imgLeft);
                normalVector nl=crossProduct(n1,n2);
                leftPlanes.push_back(n1);
                rightPlanes.push_back(n2);
                line3D l;
                l.direction[0]=nl.nx;
                l.direction[1]=nl.ny;
                l.direction[2]=nl.nz;
                l.color=(colorR[i],colorG[i],colorB[i]);
                double leftStartX=segDescLeft[matchResult[2*i]].segment.x1-(imageWidth-1)/2;
                double leftStartY=(imageHeight-1)/2-segDescLeft[matchResult[2*i]].segment.y1;
                double leftEndX=segDescLeft[matchResult[2*i]].segment.x2-(imageWidth-1)/2;
                double leftEndY=(imageHeight-1)/2-segDescLeft[matchResult[2*i]].segment.y2;
                double rightStartX=segDescRight[matchResult[2*i+1]].segment.x1-(imageWidth-1)/2+trans;
                double rightStartY=(imageHeight-1)/2-segDescRight[matchResult[2*i+1]].segment.y1;
                double rightEndX=segDescRight[matchResult[2*i+1]].segment.x2-(imageWidth-1)/2+trans;
                double rightEndY=(imageHeight-1)/2-segDescRight[matchResult[2*i+1]].segment.y2;
                double leftMidX=(leftStartX+leftEndX)/2;
                double leftMidY=(leftStartY+leftEndY)/2;
                double rightMidX=(rightStartX+rightEndX)/2;
                double rightMidY=(rightStartY+rightEndY)/2;
                l.leftEndPoint1[0]=n2.nx*trans/(n2.nx+n2.ny*leftStartY/leftStartX+n2.nz*focalLength/leftStartX);
                l.leftEndPoint1[1]=l.leftEndPoint1[0]*leftStartY/leftStartX;
                l.leftEndPoint1[2]=l.leftEndPoint1[0]*focalLength/leftStartX;
                l.leftEndPoint2[0]=n2.nx*trans/(n2.nx+n2.ny*leftEndY/leftEndX+n2.nz*focalLength/leftEndX);
                l.leftEndPoint2[1]=l.leftEndPoint2[0]*leftEndY/leftEndX;
                l.leftEndPoint2[2]=l.leftEndPoint2[0]*focalLength/leftEndX;
                l.rightEndPoint1[0]=(n1.ny*trans*rightStartY/rightStartX+n1.nz*trans*focalLength/rightStartX)/(n1.nx+n1.ny*rightStartY/rightStartX+n1.nz*focalLength/rightStartX);
                l.rightEndPoint1[1]=(l.rightEndPoint1[0]-trans)*rightStartY/rightStartX;
                l.rightEndPoint1[2]=(l.rightEndPoint1[0]-trans)*focalLength/rightStartX;
                l.rightEndPoint2[0]=(n1.ny*trans*rightEndY/rightEndX+n1.nz*trans*focalLength/rightEndX)/(n1.nx+n1.ny*rightEndY/rightEndX+n1.nz*focalLength/rightEndX);
                l.rightEndPoint2[1]=(l.rightEndPoint2[0]-trans)*rightEndY/rightEndX;
                l.rightEndPoint2[2]=(l.rightEndPoint2[0]-trans)*focalLength/rightEndX;
                l.leftMidPoint[0]=n2.nx*trans/(n2.nx+n2.ny*leftMidY/leftMidX+n2.nz*focalLength/leftMidX);
                l.leftMidPoint[1]=l.leftMidPoint[0]*leftMidY/leftMidX;
                l.leftMidPoint[2]=l.leftMidPoint[0]*focalLength/leftMidX;
                l.rightMidPoint[0]=(n1.ny*trans*rightMidY/rightMidX+n1.nz*trans*focalLength/rightMidX)/(n1.nx+n1.ny*rightMidY/rightMidX+n1.nz*focalLength/rightMidX);
                l.rightMidPoint[1]=(l.rightMidPoint[0]-trans)*rightMidY/rightMidX;
                l.rightMidPoint[2]=(l.rightMidPoint[0]-trans)*focalLength/rightMidX;
                l.ID=i;
                lineRecon.push_back(l);
 /*               out<<"Line ID"<<i+1<<":"<<endl;
                out<<"Left:"<<endl;
                out<<"Start Point:"<<l.leftEndPoint1[0]<<" "<<l.leftEndPoint1[1]<<" "<<l.leftEndPoint1[2]<<endl;
                out<<"End Point:"<<l.leftEndPoint2[0]<<" "<<l.leftEndPoint2[1]<<" "<<l.leftEndPoint2[2]<<endl;
                out<<"Mid Point:"<<l.leftMidPoint[0]<<" "<<l.leftMidPoint[1]<<" "<<l.leftMidPoint[2]<<endl;
                out<<"Left normal Vector"<<n1.nx<<" "<<n1.ny<<" "<<n1.nz<<endl;
                out<<"Right:"<<endl;
                out<<"Start Point:"<<l.rightEndPoint1[0]<<" "<<l.rightEndPoint1[1]<<" "<<l.rightEndPoint1[2]<<endl;
                out<<"End Point:"<<l.rightEndPoint2[0]<<" "<<l.rightEndPoint2[1]<<" "<<l.rightEndPoint2[2]<<endl;
                out<<"Mid Point:"<<l.rightMidPoint[0]<<" "<<l.rightMidPoint[1]<<" "<<l.rightMidPoint[2]<<endl;
                out<<"Right normal Vector"<<n2.nx<<" "<<n2.ny<<" "<<n2.nz<<endl;*/                
            }
            lineRelMat.clear();
            lineRelMat.resize(matchResult.size()/2);
            for(int i=0;i<lineRelMat.size();i++){
                lineRelMat[i].resize(matchResult.size()/2);
            }
            homographies.clear();
//            transImg.clear();
            homographies.resize(matchResult.size()/2);
//           transImg.resize(matchResult.size()/2);
            for(int i=0;i<homographies.size();i++){
                homographies[i].resize(matchResult.size()/2);
//                transImg[i].resize(matchResult.size()/2);
            }
            featuresLeft.clear();
            featuresRight.clear();
            featuresLeft.resize(homographies.size());
            featuresRight.resize(homographies.size());
            for(int i=0;i<homographies.size();i++){
                featuresLeft[i].clear();
                featuresRight[i].clear();
                Point2f tmp;
                tmp.x=segDescLeft[matchResult[2*i]].segment.x1;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y1;
                featuresLeft[i].push_back(tmp);
                tmp.x=segDescLeft[matchResult[2*i]].segment.x2;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y2;
                featuresLeft[i].push_back(tmp);
                tmp.x=(segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2;
                tmp.y=(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2;
                featuresLeft[i].push_back(tmp);
                tmp.x=(segDescLeft[matchResult[2*i]].segment.y1-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y1;
                featuresRight[i].push_back(tmp);
                tmp.x=(segDescLeft[matchResult[2*i]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y2;
                featuresRight[i].push_back(tmp);
                tmp.x=((segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2;
                featuresRight[i].push_back(tmp);
            }
//            out<<"Line Relations:"<<endl;
//            Mat pointMatch=imread("../image/right1.png");
            for(int i=0;i<lineRelMat.size();i++){
                for(int j=i+1;j<lineRelMat.size();j++){
                //    out<<"i "<<lineRecon[i].direction<<endl;
                //    out<<"j "<<lineRecon[j].direction<<endl;
                    lineRelMat[i][j].angle=getAngle(lineRecon[i],lineRecon[j]);
                    lineRelMat[j][i].angle=lineRelMat[i][j].angle;
//                    out<<i+1<<" "<<j+1<<" "<<lineRelMat[i][j].angle<<endl;
                    lineRelMat[i][j].dist=getDist(lineRecon[i],lineRecon[j]);
                    lineRelMat[j][i].dist=lineRelMat[i][j].dist;
//                    out<<lineRelMat[i][j].dist<<endl;
                    //line(pointMatch,Point(PointsRight[0].x,PointsRight[0].y),Point(PointsRight[1].x,PointsRight[1].y),Scalar(255,0,0),1);
                    //line(pointMatch,Point(PointsRight[2].x,PointsRight[2].y),Point(PointsRight[3].x,PointsRight[3].y),Scalar(255,0,0),1);
                    vector<Point2f>PointsLeft,PointsRight;
                    PointsLeft.clear();
                    PointsRight.clear();
                    PointsLeft.insert(PointsLeft.end(),featuresLeft[i].begin(),featuresLeft[i].end());
                    PointsLeft.insert(PointsLeft.end(),featuresLeft[j].begin(),featuresLeft[j].end());
                    PointsRight.insert(PointsRight.end(),featuresRight[i].begin(),featuresRight[i].end());
                    PointsRight.insert(PointsRight.end(),featuresRight[j].begin(),featuresRight[j].end());
//                    homographies[i][j]=findHomography(PointsLeft,PointsRight,RANSAC);
//                    out<<homographies[i][j]<<endl;
                    //out<<homographies[i][j].at<double>(0,0)<<" "<<homographies[i][j].at<double>(0,1)<<" "<<homographies[i][j].at<double>(0,2)<<endl;
                    //out<<homographies[i][j].at<double>(1,0)<<" "<<homographies[i][j].at<double>(1,1)<<" "<<homographies[i][j].at<double>(1,2)<<endl;
                    //out<<homographies[i][j].at<double>(2,0)<<" "<<homographies[i][j].at<double>(2,1)<<" "<<homographies[i][j].at<double>(2,2)<<endl;
//                    warpPerspective(pointMatch, transImg[i][j], homographies[i][j], Size(pointMatch.cols,pointMatch.rows));
//                    stringstream title;
//                    string str_title;
//                    title<<i+1<<"Vs"<<j+1;
//                    title>>str_title;
//                    imshow(str_title,transImg[i][j]);
//                    waitKey(0);
                }
            }
            //imshow("Match Line",pointMatch);
//            waitKey(0);
        }
        void getPlanes(){
            planeStructure.clear();
            PlaneFitEqu.clear();
            bool* checked=new bool[homographies.size()*homographies.size()];
            memset(checked,0,sizeof(bool)*homographies.size()*homographies.size());
            for(int i=0;i<homographies.size();i++){
                for(int j=i+1;j<homographies.size();j++){
                    if(checked[i*homographies.size()+j]==true)continue;
                    if(abs(lineRelMat[i][j].angle)<parallelBound && lineRelMat[i][j].dist>distBound){
                        checked[i*homographies.size()+j]=true;
                        checked[j*homographies.size()+i]=true;
                        continue;
                    }
                    if(abs(lineRelMat[i][j].angle)>parallelBound)continue;
                    vector<line3D>tmp;
                    tmp.clear();
                    tmp.push_back(lineRecon[i]);
                    tmp.push_back(lineRecon[j]);
                    checked[i*homographies.size()+j]=true;
                    checked[j*homographies.size()+i]=true;
                    Vec3f nv=getNormalVector(lineRecon[i],lineRecon[j]);
                    for(int k=0;k<homographies.size();k++){
                        bool flag=true;
                        bool cross=false;
                        if(k==i||k==j)continue;
                        if(abs(getAngle(lineRecon[k],nv))>coplanarBound)continue;
                        for(int l=0;l<tmp.size();l++){
                            if(lineRelMat[k][tmp[l].ID].dist>distBound && abs(lineRelMat[k][tmp[l].ID].angle)<parallelBound){
                                flag=false;
                                break;
                            }
                            if(abs(lineRelMat[k][tmp[l].ID].angle)<parallelBound && lineRelMat[k][tmp[l].ID].dist<distBound)cross=true;
                        }
                        if(flag&&cross){
                            for(int l=0;l<tmp.size();l++){
                                checked[k*homographies.size()+tmp[l].ID]=true;
                                checked[(tmp[l].ID)*homographies.size()+k]=true;
                            }
                            tmp.push_back(lineRecon[k]);
                        }
                    }
                    vector<int> repeatedIDs;
                    int mergeIndex=-1;
                    for(int k1=0;k1<planeStructure.size();k1++){
                        if(mergeIndex>-1)break;
                        repeatedIDs.clear();
                        for(int k2=0;k2<tmp.size();k2++){
                            if(mergeIndex>-1)break;
                            for(int k3=0;k3<planeStructure[k1].size();k3++){
                                if(tmp[k2].ID==planeStructure[k1][k3].ID){
                                    if(repeatedIDs.size()==0){
                                        repeatedIDs.push_back(tmp[k2].ID);
                                        break;
                                    }
                                    else{
                                        if((lineRelMat[repeatedIDs[0]][tmp[k2].ID].angle<parallelBound && lineRelMat[repeatedIDs[0]][tmp[k2].ID].dist<distBound)||(lineRelMat[repeatedIDs[0]][tmp[k2].ID].angle>parallelBound && lineRelMat[repeatedIDs[0]][tmp[k2].ID].dist>distBound)){
                                            repeatedIDs.push_back(tmp[k2].ID);
                                        }
                                        if(repeatedIDs.size()>1)mergeIndex=k1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if(mergeIndex>-1){
                        int formerSize=planeStructure[mergeIndex].size();
                        for(int k4=0;k4<tmp.size();k4++){
                            bool diff=false;
                            for(int k5=0;k5<formerSize;k5++){
                                if(tmp[k4].ID==planeStructure[mergeIndex][k5].ID){
                                    diff=true;
                                    break;
                                }
                            }
                            if(diff==false){
                                planeStructure[mergeIndex].push_back(tmp[k4]);
                                for(int k6=0;k6<formerSize;k6++){
                                    checked[planeStructure[mergeIndex][k6].ID*homographies.size()+tmp[k4].ID]=true;
                                    checked[(tmp[k4].ID)*homographies.size()+planeStructure[mergeIndex][k6].ID]=true;
                                }
                            }
                        }
                        float equ[4];
                        fitPlane(planeStructure[mergeIndex],equ);
                        vector<float> e;
                        e.clear();
                        e.push_back(equ[0]);
                        e.push_back(equ[1]);
                        e.push_back(equ[2]);
                        e.push_back(equ[3]);
                        PlaneFitEqu[mergeIndex]=e;
                    }
                    else{
                        planeStructure.push_back(tmp);
                        float equ[4];
                        fitPlane(planeStructure[planeStructure.size()-1],equ);
                        vector<float> e;
                        e.clear();
                        e.push_back(equ[0]);
                        e.push_back(equ[1]);
                        e.push_back(equ[2]);
                        e.push_back(equ[3]);
                        PlaneFitEqu.push_back(e);
                    }
                }
            }
            for(int i=0;i<homographies.size();i++){
                for(int j=i+1;j<homographies.size();j++){
                    if(checked[i*homographies.size()+j]==false){
                        SEGMENT test1,test2,test3,test4;
                        test1.x1=segDescLeft[matchResult[2*i]].segment.x1;
                        test1.y1=segDescLeft[matchResult[2*i]].segment.y1;
                        test1.x2=segDescLeft[matchResult[2*j]].segment.x1;
                        test1.y2=segDescLeft[matchResult[2*j]].segment.y1;
                        test2.x1=segDescLeft[matchResult[2*i]].segment.x1;
                        test2.y1=segDescLeft[matchResult[2*i]].segment.y1;
                        test2.x2=segDescLeft[matchResult[2*j]].segment.x2;
                        test2.y2=segDescLeft[matchResult[2*j]].segment.y2;
                        test3.x1=segDescLeft[matchResult[2*i]].segment.x2;
                        test3.y1=segDescLeft[matchResult[2*i]].segment.y2;
                        test3.x2=segDescLeft[matchResult[2*j]].segment.x1;
                        test3.y2=segDescLeft[matchResult[2*j]].segment.y1;
                        test4.x1=segDescLeft[matchResult[2*i]].segment.x2;
                        test4.y1=segDescLeft[matchResult[2*i]].segment.y2;
                        test4.x2=segDescLeft[matchResult[2*j]].segment.x2;
                        test4.y2=segDescLeft[matchResult[2*j]].segment.y2;
                        for(int k=0;k<homographies.size();k++){
                            if(k==i||k==j)continue;
                            if((checkIntersection(test1,segDescLeft[matchResult[2*k]].segment))||(checkIntersection(test2,segDescLeft[matchResult[2*k]].segment))||(checkIntersection(test3,segDescLeft[matchResult[2*k]].segment))||(checkIntersection(test4,segDescLeft[matchResult[2*k]].segment))){
                                checked[i*homographies.size()+j]=true;
                                checked[j*homographies.size()+i]=true;
                                break;
                            }
                        }
                    }
                }
            }
            vector<vector<line3D> >parallelPairs;
            vector<vector<float> >parallelFit;
            parallelPairs.clear();
            parallelFit.clear();
            for(int i=0;i<homographies.size();i++){
                for(int j=i+1;j<homographies.size();j++){
                    if(checked[i*homographies.size()+j]==false){
                        vector<line3D>linePairs;
                        linePairs.clear();
                        linePairs.push_back(lineRecon[i]);
                        linePairs.push_back(lineRecon[j]);
                        float fitEqu[4];
                        fitPlane(linePairs,fitEqu);
                        bool checkMerge=false;
                        for(int k=0;k<parallelPairs.size();k++){
                            if(checkMerge)break;
                            bool coincidenceI=false;
                            bool coincidenceJ=false;
                            for(int l=0;l<parallelPairs[k].size();l++){
                                if(parallelPairs[k][l].ID==i && abs(parallelFit[k][0]*fitEqu[0]+parallelFit[k][1]*fitEqu[1]+parallelFit[k][2]*fitEqu[2])>parallelBound)coincidenceI=true;
                                if(parallelPairs[k][l].ID==j && abs(parallelFit[k][0]*fitEqu[0]+parallelFit[k][1]*fitEqu[1]+parallelFit[k][2]*fitEqu[2])>parallelBound)coincidenceJ=true;
                                if(coincidenceI && coincidenceJ){
                                    checkMerge=true;
                                    break;
                                }
                            }
                            if(coincidenceI && ! coincidenceJ){
                                parallelPairs[k].push_back(lineRecon[j]);
                                float equ1[4];
                                fitPlane(parallelPairs[k],equ1);
                                vector<float> fitEqu1;
                                fitEqu1.clear();
                                fitEqu1.push_back(equ1[0]);
                                fitEqu1.push_back(equ1[1]);
                                fitEqu1.push_back(equ1[2]);
                                fitEqu1.push_back(equ1[3]);
                                parallelFit[k]=fitEqu1;
                                checkMerge=true;
                                break;
                            }
                            else if(coincidenceJ && ! coincidenceI){
                                parallelPairs[k].push_back(lineRecon[i]);
                                float equ2[4];
                                fitPlane(parallelPairs[k],equ2);
                                vector<float> fitEqu2;
                                fitEqu2.clear();
                                fitEqu2.push_back(equ2[0]);
                                fitEqu2.push_back(equ2[1]);
                                fitEqu2.push_back(equ2[2]);
                                fitEqu2.push_back(equ2[3]);
                                parallelFit[k]=fitEqu2;
                                checkMerge=true;
                                break;
                            }
                        }
                        if(!checkMerge){
                            vector<float> coef;
                            coef.clear();
                            coef.push_back(fitEqu[0]);
                            coef.push_back(fitEqu[1]);
                            coef.push_back(fitEqu[2]);
                            coef.push_back(fitEqu[3]);
                            parallelPairs.push_back(linePairs);
                            parallelFit.push_back(coef);
                        }
                    }
                }
            }
            for(int i=0;i<parallelPairs.size();i++){
                planeStructure.push_back(parallelPairs[i]);
                PlaneFitEqu.push_back(parallelFit[i]);
            }
            delete[] checked;
            PlaneHomographies.clear();
            PlaneHomographies.resize(planeStructure.size());
            constraintH.resize(planeStructure.size());
 //           out<<"Planes found:"<<planeStructure.size()<<endl;
            for(int i=0;i<planeStructure.size();i++){
                Mat tmp;
                tmp=imread(fLeft);
                vector<Point2f>PointsLeft,PointsRight;
                PointsLeft.clear();
                PointsRight.clear();
                //tmp=imread("neem1.png");
                for(int j=0;j<planeStructure[i].size();j++){
 //                   out<<planeStructure[i][j].ID+1<<" ";
 //                   line(tmp,Point(segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.x1,segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.y1),Point(segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.x2,segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.y2),Scalar(255,0,0),3);
                    PointsLeft.insert(PointsLeft.end(),featuresLeft[planeStructure[i][j].ID].begin(),featuresLeft[planeStructure[i][j].ID].end());
                    PointsRight.insert(PointsRight.end(),featuresRight[planeStructure[i][j].ID].begin(),featuresRight[planeStructure[i][j].ID].end());
                }
                Mat hP;
                float Hx[4];
                Mat HH;
                hP.create(PointsLeft.size(),3,CV_32FC1);
                for(int zz=0;zz<PointsLeft.size();zz++){
                    hP.at<float>(zz,0)=PointsLeft[zz].x;
                    hP.at<float>(zz,1)=PointsLeft[zz].y;
                    hP.at<float>(zz,2)=PointsRight[zz].x;
                }
                cvFitPlane(hP,Hx);
                HH.create(3,3,CV_32FC1);
                HH.at<float>(0,0)=-Hx[0]/Hx[2];
                HH.at<float>(0,1)=-Hx[1]/Hx[2];
                HH.at<float>(0,2)=Hx[3]/Hx[2];
                HH.at<float>(1,0)=0;
                HH.at<float>(1,1)=1;
                HH.at<float>(1,2)=0;
                HH.at<float>(2,0)=0;
                HH.at<float>(2,1)=0;
                HH.at<float>(2,2)=1;
//                out<<endl;
//                out<<PlaneFitEqu[i][0]<<" "<<PlaneFitEqu[i][1]<<" "<<PlaneFitEqu[i][2]<<" "<<PlaneFitEqu[i][3]<<endl;
                PlaneHomographies[i]=findHomography(PointsLeft,PointsRight,RANSAC);
                constraintH[i]=HH;
//                out<<PlaneHomographies[i]<<endl;
//                imshow("",tmp);
//                Mat pointMatch=imread(fLeft);
//                Mat transImg;
//                Mat transImg2;
//                Mat H=(Mat_<float>(3,3)<<(-PlaneFitEqu[i][3]-trans*PlaneFitEqu[i][0])/(-PlaneFitEqu[i][3]),-trans*PlaneFitEqu[i][1]/(-PlaneFitEqu[i][3]),-trans*PlaneFitEqu[i][2]*pixelZ/(-PlaneFitEqu[i][3]),0,-PlaneFitEqu[i][3]/(-PlaneFitEqu[i][3]),0,0,0,-PlaneFitEqu[i][3]/(-PlaneFitEqu[i][3]));
//                out<<H<<endl;
//                out<<HH<<endl;
/*                double di=trans/(HH.at<float>(0,0)-1);
                double n1=1.0;
                double n2=HH.at<float>(0,1)*di/trans;
                double n3=HH.at<float>(0,2)*di/(trans*pixelZ);
                double rr=sqrt(n1*n1+n2*n2+n3*n3);
                n1=n1/rr;
                n2=n2/rr;
                n3=n3/rr;
                di=di/rr;
                cout<<n1<<" "<<n2<<" "<<n3<<" "<<di<<endl;
                warpPerspective(pointMatch, transImg, PlaneHomographies[i], Size(pointMatch.cols,pointMatch.rows));
                warpPerspective(pointMatch, transImg2, HH, Size(pointMatch.cols,pointMatch.rows));
                imshow("trans",transImg);
                imshow("epi",transImg2);
                stringstream fn;
                string ttl;
                fn<<"Plane"<<i+1<<".png";
                fn>>ttl;
                imwrite(ttl,tmp);
                waitKey(0);*/
            }
            for(int i=0;i<PlaneHomographies.size();i++){
                PlaneHomographies[i].convertTo(PlaneHomographies[i],CV_32FC1);
            }
            for(int i=1;i<constraintH.size();i++){
                for(int j=0;j<i;j++){
                    double p0,p1,p2,p3;
                    getCoef(constraintH[i],constraintH[j],p0,p1,p2,p3);
                    out<<endl;
                    out<<i<<" "<<j<<endl;
                    out<<discriminant(p0,p1,p2,p3)<<endl;
                    getCoef(PlaneHomographies[i],PlaneHomographies[j],p0,p1,p2,p3);
                    out<<endl;
                    out<<i<<" "<<j<<endl;
                    out<<discriminant(p0,p1,p2,p3)<<endl;
                }
            }
            for(int i=1;i<constraintH.size();i++){
                for(int j=0;j<i;j++){
                    for(int k=1;k<constraintH.size();k++){
                        for(int l=0;l<k;l++){
                            out<<i<<" "<<j<<" "<<k<<" "<<l<<endl;
                            for(int m=1;m<4;m++){
                                for(int n=1;n<4;n++){
                                    out<<m<<" "<<n<<endl;
                                    out<<LieBracket(constraintH[i],constraintH[j],constraintH[k],constraintH[l],m,n)<<endl;
                                    out<<LieBracket(PlaneHomographies[i],PlaneHomographies[j],PlaneHomographies[k],PlaneHomographies[l],m,n)<<endl;
                                }
                            }
                        }
                    }
                }
            }
        }
};
void CalcH(const string filename1,const string filename2){
    ofstream wt;
    ofstream hg;
    wt.open("lineMatches.txt");
    hg.open("homographies.txt");
    vector<double>lx1,ly1,lx2,ly2,rx1,ry1,rx2,ry2;
    vector<double>lmx,lmy,rmx,rmy;
    lx1.clear();ly1.clear();lx2.clear();ly2.clear();rx1.clear();ry1.clear();rx2.clear();ry2.clear();lmx.clear();lmy.clear();rmx.clear();rmy.clear();
    Mat imgLeft=imread(filename1);
    Mat imgRight=imread(filename2);
    Mat imgLeftGray,imgRightGray;
    vector<SEGMENT>linesLeft,linesRight;
    vector<segDesc> segDescLeft,segDescRight;
    LineDetector ld;
    vector<unsigned int> matchResult;
    vector<double>colorR,colorG,colorB;
    int imageHeight,imageWidth;
    vector<Point2f>PointsLeft,PointsRight;
    vector<Mat> H1,H2;
    cvtColor(imgLeft,imgLeftGray,COLOR_RGB2GRAY);
    cvtColor(imgRight,imgRightGray,COLOR_RGB2GRAY);
    imageWidth=imgLeft.cols;
    imageHeight=imgLeft.rows;
    linesLeft.clear();
    linesRight.clear();
    ld.lineDetection(imgLeftGray,linesLeft);
    ld.lineDetection(imgRightGray,linesRight);
    descriptor desc(9,7);
    desc.getGradientMap(filename1);
    segDescLeft.clear();
    segDescRight.clear();
	segDescLeft.resize(linesLeft.size());
  	segDescRight.resize(linesRight.size());
	for(int i=0;i<linesLeft.size();i++){
	    segDescLeft[i].segment.x1=linesLeft.at(i).x1;
  		segDescLeft[i].segment.x2=linesLeft.at(i).x2;
	  	segDescLeft[i].segment.y1=linesLeft.at(i).y1;
		segDescLeft[i].segment.y2=linesLeft.at(i).y2;
  		segDescLeft[i].segment.angle=linesLeft.at(i).angle;
  		segDescLeft[i].segment.label=i;
  		segDescLeft[i].lineLength=sqrt((linesLeft.at(i).x2-linesLeft.at(i).x1)*(linesLeft.at(i).x2-linesLeft.at(i).x1)+(linesLeft.at(i).y2-linesLeft.at(i).y1)*(linesLeft.at(i).y2-linesLeft.at(i).y1));
  		segDescLeft[i].desVec=desc.calcLBD(linesLeft.at(i));
  	}
  	desc.getGradientMap(filename2);
  	for(int i=0;i<linesRight.size();i++){
  		segDescRight[i].segment.x1=linesRight.at(i).x1;
  		segDescRight[i].segment.x2=linesRight.at(i).x2;
  		segDescRight[i].segment.y1=linesRight.at(i).y1;
  		segDescRight[i].segment.y2=linesRight.at(i).y2;
  		segDescRight[i].segment.angle=linesRight.at(i).angle;
  		segDescRight[i].segment.label=i;
  		segDescRight[i].lineLength=sqrt((linesRight.at(i).x2-linesRight.at(i).x1)*(linesRight.at(i).x2-linesRight.at(i).x1)+(linesRight.at(i).y2-linesRight.at(i).y1)*(linesRight.at(i).y2-linesRight.at(i).y1));
  		segDescRight[i].desVec=desc.calcLBD(linesRight.at(i));
  	}
	BuildMat(segDescLeft,segDescRight);
    matchResult.clear();
	MatchingResult(segDescLeft,segDescRight,matchResult);
    srand((int)time(0));
    colorR.clear();
    colorG.clear();
    colorB.clear();
    colorR.resize(matchResult.size()/2);
    colorG.resize(matchResult.size()/2);
    colorB.resize(matchResult.size()/2);
    for(int i=0;i<matchResult.size()/2;i++){
        colorR[i]=((double)rand())/RAND_MAX;
        colorG[i]=((double)rand())/RAND_MAX;
        colorB[i]=((double)rand())/RAND_MAX;
    }
    Mat img=imread(filename1);
    Mat img2=imread(filename2);
    string iid;
    for(unsigned int i=0;i<matchResult.size()/2;i++){
        stringstream sss;
        sss<<i+1;
        sss>>iid;
        line(img,Point(segDescLeft[matchResult[2*i]].segment.x1,segDescLeft[matchResult[2*i]].segment.y1),Point(segDescLeft[matchResult[2*i]].segment.x2,segDescLeft[matchResult[2*i]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
        wt<<i+1<<endl;
        wt<<segDescLeft[matchResult[2*i]].segment.x1<<" "<<segDescLeft[matchResult[2*i]].segment.y1<<" "<<segDescLeft[matchResult[2*i]].segment.x2<<" "<<segDescLeft[matchResult[2*i]].segment.y2<<endl;
        lx1.push_back(segDescLeft[matchResult[2*i]].segment.x1);
        ly1.push_back(segDescLeft[matchResult[2*i]].segment.y1);
        lx2.push_back(segDescLeft[matchResult[2*i]].segment.x2);
        ly2.push_back(segDescLeft[matchResult[2*i]].segment.y2);
        lmx.push_back((segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2);
        lmy.push_back((segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2);
        putText(img,iid,Point((segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2,(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
        line(img2,Point(segDescRight[matchResult[2*i+1]].segment.x1,segDescRight[matchResult[2*i+1]].segment.y1),Point(segDescRight[matchResult[2*i+1]].segment.x2,segDescRight[matchResult[2*i+1]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
        wt<<segDescRight[matchResult[2*i+1]].segment.x1<<" "<<segDescRight[matchResult[2*i+1]].segment.y1<<" "<<segDescRight[matchResult[2*i+1]].segment.x2<<" "<<segDescRight[matchResult[2*i+1]].segment.y2<<endl;
        rx1.push_back(segDescRight[matchResult[2*i+1]].segment.x1);
        ry1.push_back(segDescRight[matchResult[2*i+1]].segment.y1);
        rx2.push_back(segDescRight[matchResult[2*i+1]].segment.x2);
        ry2.push_back(segDescRight[matchResult[2*i+1]].segment.y2);
        rmx.push_back((segDescRight[matchResult[2*i+1]].segment.x1+segDescRight[matchResult[2*i+1]].segment.x2)/2);
        rmy.push_back((segDescRight[matchResult[2*i+1]].segment.y1+segDescRight[matchResult[2*i+1]].segment.y2)/2);
        putText(img2,iid,Point((segDescRight[matchResult[2*i+1]].segment.x1+segDescRight[matchResult[2*i+1]].segment.x2)/2,(segDescRight[matchResult[2*i+1]].segment.y1+segDescRight[matchResult[2*i+1]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
    }
    imshow("Left View",img);
    imshow("Right View",img2);
    imwrite("outleft.png",img);
    imwrite("outright.png",img2);
    waitKey(0);
    Mat planar1,planar2;
    planar1=imread(filename1);
    planar2=imread(filename2);
    int p,q;
    H1.clear();H2.clear();
    while(1){
        cout<<"p:"<<endl;
        cin>>p;
        if(p==-1)break;
        PointsLeft.clear();
        PointsRight.clear();
        while(1){
            cin>>q;
            if(q==-1)break;
            Point2f tmp;
            tmp.x=lx1[q-1];
            tmp.y=ly1[q-1];
            PointsLeft.push_back(tmp);
            tmp.x=lx2[q-1];
            tmp.y=ly2[q-1];
            PointsLeft.push_back(tmp);
            tmp.x=lmx[q-1];
            tmp.y=lmy[q-1];
            PointsLeft.push_back(tmp);
            tmp.x=(ly1[q-1]-ry2[q-1])*(rx2[q-1]-rx1[q-1])/(ry2[q-1]-ry1[q-1])+rx2[q-1];
            tmp.y=ly1[q-1];
            PointsRight.push_back(tmp);
            tmp.x=(ly2[q-1]-ry2[q-1])*(rx2[q-1]-rx1[q-1])/(ry2[q-1]-ry1[q-1])+rx2[q-1];
            tmp.y=ly2[q-1];
            PointsRight.push_back(tmp);
            tmp.x=((ly1[q-1]+ly2[q-1])/2-ry2[q-1])*(rx2[q-1]-rx1[q-1])/(ry2[q-1]-ry1[q-1])+rx2[q-1];
            tmp.y=(ly1[q-1]+ly2[q-1])/2;
            PointsRight.push_back(tmp);
            line(planar1,Point(lx1[q-1],ly1[q-1]),Point(lx2[q-1],ly2[q-1]),Scalar(colorB[p-1]*255,colorG[p-1]*255,colorR[p-1]*255),3);
            line(planar2,Point(rx1[q-1],ry1[q-1]),Point(rx2[q-1],ry2[q-1]),Scalar(colorB[p-1]*255,colorG[p-1]*255,colorR[p-1]*255),3);
        }
        H1.push_back(findHomography(PointsLeft,PointsRight,RANSAC));
        Mat hP;
        float Hx[4];
        Mat HH;
        hP.create(PointsLeft.size(),3,CV_32FC1);
        for(int zz=0;zz<PointsLeft.size();zz++){
            hP.at<float>(zz,0)=PointsLeft[zz].x;
            hP.at<float>(zz,1)=PointsLeft[zz].y;
            hP.at<float>(zz,2)=PointsRight[zz].x;
        }
        cvFitPlane(hP,Hx);
        HH.create(3,3,CV_32FC1);
        HH.at<float>(0,0)=-Hx[0]/Hx[2];
        HH.at<float>(0,1)=-Hx[1]/Hx[2];
        HH.at<float>(0,2)=Hx[3]/Hx[2];
        HH.at<float>(1,0)=0;
        HH.at<float>(1,1)=1;
        HH.at<float>(1,2)=0;
        HH.at<float>(2,0)=0;
        HH.at<float>(2,1)=0;
        HH.at<float>(2,2)=1;
        H2.push_back(HH);
    }
    double cons1=0.0;
    double cons2=0.0;
    double cons3=0.0;
    double cons4=0.0;
    double cons5=0.0;
    int cnt1=0;
    int cnt2=0;
    double val;
    for(int i=0;i<H1.size();i++){
        H1[i].convertTo(H1[i],CV_32FC1);
    }
    for(int i=1;i<H2.size();i++){
        for(int j=0;j<i;j++){
            double p0,p1,p2,p3;
            getCoef(H2[i],H2[j],p0,p1,p2,p3);
            hg<<endl;
            hg<<i<<" "<<j<<endl;
            val=abs(discriminant(p0,p1,p2,p3));
            hg<<val<<endl;
            cons1+=val;
            getCoef(H1[i],H1[j],p0,p1,p2,p3);
            hg<<endl;
            hg<<i<<" "<<j<<endl;
            val=abs(discriminant(p0,p1,p2,p3));
            hg<<val<<endl;
            cons2+=val;
            cnt1++;
        }
    }
    cons1/=cnt1;
    cons2/=cnt1;
    for(int i=1;i<H2.size();i++){
        for(int j=0;j<i;j++){
            for(int k=1;k<H2.size();k++){
                for(int l=0;l<k;l++){
                    hg<<i<<" "<<j<<" "<<k<<" "<<l<<endl;
                    for(int m=1;m<4;m++){
                        for(int n=1;n<4;n++){
                            hg<<m<<" "<<n<<endl;
                            val=LieBracket(H2[i],H2[j],H2[k],H2[l],m,n);
                            hg<<val<<endl;
                            cons3+=abs(val);
                            val=LieBracket(H1[i],H1[j],H1[k],H1[l],m,n);
                            hg<<val<<endl;
                            cons4+=abs(val);
                            cons5+=val;
                            cnt2++;
                        }
                    }
                }
            }
        }
    }
    ofstream result;
    result.open("eval.txt");
    result<<cons1<<" "<<cons2<<" "<<cons3<<" "<<cons4<<" "<<cons5<<" "<<cons3/cnt2<<" "<<cons4/cnt2<<" "<<cons5/cnt2<<endl;
    imshow("plane1",planar1);
    imshow("plane2",planar2);
    imwrite("leftPlane.png",planar1);
    imwrite("rightPlane.png",planar2);
}

double stereo::focalLength=438.4;
Vec3f stereo::translation=(4.5,0.0,0.0);
stereo s;
void init(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH,GL_NICEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}
void display(void){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glRotatef(rtri,0.0f,1.0f,0.0f);
    for(int i=0;i<s.matchResult.size()/2;i++){
        glBegin(GL_LINES);
        glColor3f(s.colorR[i],s.colorG[i],s.colorB[i]);
        glVertex3f(s.lineRecon[i].leftEndPoint1[0]/35-1,s.lineRecon[i].leftEndPoint1[1]/35,4-s.lineRecon[i].leftEndPoint1[2]/35);
        glVertex3f(s.lineRecon[i].leftEndPoint2[0]/35-1,s.lineRecon[i].leftEndPoint2[1]/35,4-s.lineRecon[i].leftEndPoint2[2]/35);
        glEnd();
        glFlush();
/*      glBegin(GL_LINES);
        glColor3f(s.colorR[i],s.colorG[i],s.colorB[i]);
        glVertex3f(s.lineRecon[i].rightEndPoint1[0]/15-2,s.lineRecon[i].rightEndPoint1[1]/15,s.lineRecon[i].rightEndPoint1[2]/15-2);
        glVertex3f(s.lineRecon[i].rightEndPoint2[0]/15-2,s.lineRecon[i].rightEndPoint2[1]/15,s.lineRecon[i].rightEndPoint2[2]/15-2);
        glEnd();
        glFlush();*/
    }
    glutSwapBuffers();
    rtri+=0.3;
}
void reshape (int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 1.0, 100.0);
    gluLookAt(0, 0, 5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'x':
        case 27: 
        exit(0);
        break;
        default:
        break;
    }
}

void rsim(int cnt,vector<vector<double> >& Point3D,vector<vector<double> >& Point2D,double err=0.0){
    double f=438.4;
    double t=4.5/438.4;
    double x0=320;
    double y0=256;
    Point3D.resize(12);
    Point2D.resize(16);
    vector<vector<double> >error;
    error.resize(16);
    for(int i=0;i<Point3D.size();i++){
        Point3D[i].resize(cnt);
    }
    for(int i=0;i<Point2D.size();i++){
        Point2D[i].resize(cnt);
        error[i].resize(cnt);
    }
    for(int i=0;i<cnt;i++){
        Point3D[0][i]=-5;
        Point3D[1][i]=3;
        Point3D[2][i]=2;
        Point3D[3][i]=0;
        Point3D[4][i]=4;
        Point3D[5][i]=5;
        Point3D[6][i]=5;
        Point3D[7][i]=3;
        Point3D[8][i]=2;
        Point3D[9][i]=1;
        Point3D[10][i]=0;
        Point3D[11][i]=4;
    }
    srand((int)time(0));
    for(int i=0;i<error.size();i++){
        for(int j=0;j<cnt;j++){
            error[i][j]=(double)rand()/RAND_MAX*2*err-err;
        }
    }
    double val1=0,val2=0,val3=0,val4=0;
    for(int i=0;i<cnt;i++){
        vector<Point2f> left,right;
        Mat hP;
        float Hx[4];
        Mat H21,H22,H23;
        hP.create(5,3,CV_32FC1);
        left.clear();right.clear();
        Point2D[0][i]=f*Point3D[0][i]/Point3D[2][i]+x0+error[0][i];
        Point2D[1][i]=f*Point3D[1][i]/Point3D[2][i]+y0+error[1][i];
        Point2D[2][i]=f*Point3D[3][i]/Point3D[5][i]+x0+error[2][i];
        Point2D[3][i]=f*Point3D[4][i]/Point3D[5][i]+y0+error[3][i];
        Point2D[4][i]=f*Point3D[6][i]/Point3D[8][i]+x0+error[4][i];
        Point2D[5][i]=f*Point3D[7][i]/Point3D[8][i]+y0+error[5][i];
        Point2D[6][i]=f*Point3D[9][i]/Point3D[11][i]+x0+error[6][i];
        Point2D[7][i]=f*Point3D[10][i]/Point3D[11][i]+y0+error[7][i];
        Point2D[8][i]=f*(Point3D[0][i]-t)/Point3D[2][i]+x0+error[8][i];
        Point2D[9][i]=f*Point3D[1][i]/Point3D[2][i]+y0+error[9][i];
        Point2D[10][i]=f*(Point3D[3][i]-t)/Point3D[5][i]+x0+error[10][i];
        Point2D[11][i]=f*Point3D[4][i]/Point3D[5][i]+y0+error[11][i];
        Point2D[12][i]=f*(Point3D[6][i]-t)/Point3D[8][i]+x0+error[12][i];
        Point2D[13][i]=f*Point3D[7][i]/Point3D[8][i]+y0+error[13][i];
        Point2D[14][i]=f*(Point3D[9][i]-t)/Point3D[11][i]+x0+error[14][i];
        Point2D[15][i]=f*Point3D[10][i]/Point3D[11][i]+y0+error[15][i];
        Point2f tmp;
        tmp.x=Point2D[0][i];
        tmp.y=Point2D[1][i];
        left.push_back(tmp);
        tmp.x=Point2D[2][i];
        tmp.y=Point2D[3][i];
        left.push_back(tmp);
        tmp.x=Point2D[4][i];
        tmp.y=Point2D[5][i];
        left.push_back(tmp);
        tmp.x=(Point2D[0][i]+Point2D[2][i])/2;
        tmp.y=(Point2D[1][i]+Point2D[3][i])/2;
        left.push_back(tmp);
        tmp.x=(Point2D[2][i]+Point2D[4][i])/2;
        tmp.y=(Point2D[3][i]+Point2D[5][i])/2;
        left.push_back(tmp);
        tmp.x=Point2D[8][i];
        tmp.y=Point2D[9][i];
        right.push_back(tmp);
        tmp.x=Point2D[10][i];
        tmp.y=Point2D[11][i];
        right.push_back(tmp);
        tmp.x=Point2D[12][i];
        tmp.y=Point2D[13][i];
        right.push_back(tmp);
        tmp.x=(Point2D[8][i]+Point2D[10][i])/2;
        tmp.y=(Point2D[9][i]+Point2D[11][i])/2;
        right.push_back(tmp);
        tmp.x=(Point2D[10][i]+Point2D[12][i])/2;
        tmp.y=(Point2D[11][i]+Point2D[13][i])/2;
        right.push_back(tmp);
        Mat H11=findHomography(left,right,RANSAC);
        for(int zz=0;zz<5;zz++){
            hP.at<float>(zz,0)=left[zz].x;
            hP.at<float>(zz,1)=left[zz].y;
            hP.at<float>(zz,2)=right[zz].x;
        }
        cvFitPlane(hP,Hx);
        H21.create(3,3,CV_32FC1);
        H21.at<float>(0,0)=-Hx[0]/Hx[2];
        H21.at<float>(0,1)=-Hx[1]/Hx[2];
        H21.at<float>(0,2)=Hx[3]/Hx[2];
        H21.at<float>(1,0)=0;
        H21.at<float>(1,1)=1;
        H21.at<float>(1,2)=0;
        H21.at<float>(2,0)=0;
        H21.at<float>(2,1)=0;
        H21.at<float>(2,2)=1;
        left.clear();right.clear();
        tmp.x=Point2D[0][i];
        tmp.y=Point2D[1][i];
        left.push_back(tmp);
        tmp.x=Point2D[2][i];
        tmp.y=Point2D[3][i];
        left.push_back(tmp);
        tmp.x=Point2D[6][i];
        tmp.y=Point2D[7][i];
        left.push_back(tmp);
        tmp.x=(Point2D[0][i]+Point2D[2][i])/2;
        tmp.y=(Point2D[1][i]+Point2D[3][i])/2;
        left.push_back(tmp);
        tmp.x=(Point2D[2][i]+Point2D[6][i])/2;
        tmp.y=(Point2D[3][i]+Point2D[7][i])/2;
        left.push_back(tmp);
        tmp.x=Point2D[8][i];
        tmp.y=Point2D[9][i];
        right.push_back(tmp);
        tmp.x=Point2D[10][i];
        tmp.y=Point2D[11][i];
        right.push_back(tmp);
        tmp.x=Point2D[14][i];
        tmp.y=Point2D[15][i];
        right.push_back(tmp);
        tmp.x=(Point2D[8][i]+Point2D[10][i])/2;
        tmp.y=(Point2D[9][i]+Point2D[11][i])/2;
        right.push_back(tmp);
        tmp.x=(Point2D[10][i]+Point2D[14][i])/2;
        tmp.y=(Point2D[11][i]+Point2D[15][i])/2;
        right.push_back(tmp);
        Mat H12=findHomography(left,right,RANSAC);
        for(int zz=0;zz<5;zz++){
            hP.at<float>(zz,0)=left[zz].x;
            hP.at<float>(zz,1)=left[zz].y;
            hP.at<float>(zz,2)=right[zz].x;
        }
        cvFitPlane(hP,Hx);
        H22.create(3,3,CV_32FC1);
        H22.at<float>(0,0)=-Hx[0]/Hx[2];
        H22.at<float>(0,1)=-Hx[1]/Hx[2];
        H22.at<float>(0,2)=Hx[3]/Hx[2];
        H22.at<float>(1,0)=0;
        H22.at<float>(1,1)=1;
        H22.at<float>(1,2)=0;
        H22.at<float>(2,0)=0;
        H22.at<float>(2,1)=0;
        H22.at<float>(2,2)=1;
        left.clear();right.clear();
        tmp.x=Point2D[2][i];
        tmp.y=Point2D[3][i];
        left.push_back(tmp);
        tmp.x=Point2D[4][i];
        tmp.y=Point2D[5][i];
        left.push_back(tmp);
        tmp.x=Point2D[6][i];
        tmp.y=Point2D[7][i];
        left.push_back(tmp);
        tmp.x=(Point2D[2][i]+Point2D[4][i])/2;
        tmp.y=(Point2D[3][i]+Point2D[5][i])/2;
        left.push_back(tmp);
        tmp.x=(Point2D[2][i]+Point2D[6][i])/2;
        tmp.y=(Point2D[3][i]+Point2D[7][i])/2;
        left.push_back(tmp);
        tmp.x=Point2D[10][i];
        tmp.y=Point2D[11][i];
        right.push_back(tmp);
        tmp.x=Point2D[12][i];
        tmp.y=Point2D[13][i];
        right.push_back(tmp);
        tmp.x=Point2D[14][i];
        tmp.y=Point2D[15][i];
        right.push_back(tmp);
        tmp.x=(Point2D[10][i]+Point2D[12][i])/2;
        tmp.y=(Point2D[11][i]+Point2D[13][i])/2;
        right.push_back(tmp);
        tmp.x=(Point2D[10][i]+Point2D[14][i])/2;
        tmp.y=(Point2D[11][i]+Point2D[15][i])/2;
        right.push_back(tmp);
        Mat H13=findHomography(left,right,RANSAC);
        for(int zz=0;zz<5;zz++){
            hP.at<float>(zz,0)=left[zz].x;
            hP.at<float>(zz,1)=left[zz].y;
            hP.at<float>(zz,2)=right[zz].x;
        }
        cvFitPlane(hP,Hx);
        H23.create(3,3,CV_32FC1);
        H23.at<float>(0,0)=-Hx[0]/Hx[2];
        H23.at<float>(0,1)=-Hx[1]/Hx[2];
        H23.at<float>(0,2)=Hx[3]/Hx[2];
        H23.at<float>(1,0)=0;
        H23.at<float>(1,1)=1;
        H23.at<float>(1,2)=0;
        H23.at<float>(2,0)=0;
        H23.at<float>(2,1)=0;
        H23.at<float>(2,2)=1;
        left.clear();right.clear();
        H11.convertTo(H11,CV_32FC1);
        H12.convertTo(H12,CV_32FC1);
        H13.convertTo(H13,CV_32FC1);
        double p0,p1,p2,p3;
        getCoef(H11,H12,p0,p1,p2,p3);
        val1+=abs(discriminant(p0,p1,p2,p3));
        getCoef(H11,H13,p0,p1,p2,p3);
        val1+=abs(discriminant(p0,p1,p2,p3));
        getCoef(H12,H13,p0,p1,p2,p3);
        val1+=abs(discriminant(p0,p1,p2,p3));
        getCoef(H21,H22,p0,p1,p2,p3);
        val2+=abs(discriminant(p0,p1,p2,p3));
        getCoef(H21,H23,p0,p1,p2,p3);
        val2+=abs(discriminant(p0,p1,p2,p3));
        getCoef(H22,H23,p0,p1,p2,p3);
        val2+=abs(discriminant(p0,p1,p2,p3));
        val3+=abs(LieBracket(H11,H12,H11,H13,3,3));
        val3+=abs(LieBracket(H11,H12,H12,H11,3,3));
        val3+=abs(LieBracket(H11,H12,H12,H13,3,3));
        val3+=abs(LieBracket(H11,H12,H13,H11,3,3));
        val3+=abs(LieBracket(H11,H12,H13,H12,3,3));
        val4+=abs(LieBracket(H21,H22,H21,H23,3,3));
        val4+=abs(LieBracket(H21,H22,H22,H21,3,3));
        val4+=abs(LieBracket(H21,H22,H22,H23,3,3));
        val4+=abs(LieBracket(H21,H22,H23,H21,3,3));
        val4+=abs(LieBracket(H21,H22,H23,H22,3,3));
    }
    val1/=3*cnt;
    val2/=3*cnt;
    val3/=5*cnt;
    val4/=5*cnt;
    sim<<err<<endl<<val1<<endl<<val2<<endl<<val3<<endl<<val4<<endl;
    cout<<val1<<endl<<val2<<endl<<val3<<endl<<val4<<endl;
}

int main(int argc, char **argv){
/*    sim.open("sim.txt");
    vector<vector<double> >t,d;
    for(double e=0.0;e<100;e+=1){
        cout<<e<<":"<<endl;
        rsim(1000,t,d,e);
    }*/
//    CalcH(argv[1],argv[2]);
    double t1 = (double)getTickCount();
    out.open("info.txt");
    cout<<argv[1];
    cout<<argv[2];
    fLeft=argv[1];
    fRight=argv[2];
    s.set(fLeft,fRight);
    double t2 = (double)getTickCount()-t1;
    cout<<t2/getTickFrequency()<<endl;
//    s.set("neem1.png","neem2.png");
    s.getPlanes();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Line Reconstruction");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(display);
    glutMainLoop();
    double t3 = (double)getTickCount()-t1;
    cout<<t3/getTickFrequency()<<endl;
/*    Mat imgLeft=imread(argv[1]);
    Mat imgRight=imread(argv[2]);
	Mat imgLeftResult=imread(argv[1]);
	Mat imgRightResult=imread(argv[2]);
    Mat im1=imread(argv[1]);
    Mat im2=imread(argv[2]);
    cvtColor(im1,im1,COLOR_RGB2GRAY);
    cvtColor(im2,im2,COLOR_RGB2GRAY);
    Mat imgLeftGray,imgRightGray;
    cvtColor(imgLeft,imgLeftGray,COLOR_RGB2GRAY);
    cvtColor(imgRight,imgRightGray,COLOR_RGB2GRAY);
    LineDetector ld;
    vector<SEGMENT>lines1,lines2;
    lines1.clear();
    lines2.clear();
    ld.lineDetection(imgLeftGray,lines1);
    ld.lineDetection(imgRightGray,lines2);
    for(size_t i=0; i<lines1.size(); i++) {
        SEGMENT seg=lines1.at(i);
        int b = (seg.label*12337) % 256;
        int g = (seg.label*24776) % 256;
        int r = (seg.label*11491) % 256;
        ld.drawArrow(imgLeftResult, &seg, Scalar(b,g,r));
    }
    for(size_t j=0; j<lines2.size(); j++) {
        SEGMENT seg=lines2.at(j);
        int b = (seg.label*12337) % 256;
        int g = (seg.label*24776) % 256;
        int r = (seg.label*11491) % 256;
        ld.drawArrow(imgRightResult, &seg, Scalar(b,g,r));
    }
    descriptor desc(9,7);
    desc.getGradientMap(argv[1]);
	vector<segDesc> segDescLeft,segDescRight;
	segDescLeft.resize(lines1.size());
  	segDescRight.resize(lines2.size());
	for(int i=0;i<lines1.size();i++){
	  	segDescLeft[i].segment.x1=lines1.at(i).x1;
  		segDescLeft[i].segment.x2=lines1.at(i).x2;
	  	segDescLeft[i].segment.y1=lines1.at(i).y1;
		  segDescLeft[i].segment.y2=lines1.at(i).y2;
  		segDescLeft[i].segment.angle=lines1.at(i).angle;
  		segDescLeft[i].segment.label=i;
  		segDescLeft[i].lineLength=sqrt((lines1.at(i).x2-lines1.at(i).x1)*(lines1.at(i).x2-lines1.at(i).x1)+(lines1.at(i).y2-lines1.at(i).y1)*(lines1.at(i).y2-lines1.at(i).y1));
  		segDescLeft[i].desVec=desc.calcLBD(lines1.at(i));
  		cout<<"LineID"<<segDescLeft[i].segment.label<<endl;
  		for(int j=0;j<segDescLeft[i].desVec.size();j++){
  			cout<<segDescLeft[i].desVec[j]<<" ";
  		}
  		cout<<endl;
  	}
  	desc.getGradientMap(argv[2]);
  	for(int i=0;i<lines2.size();i++){
  		segDescRight[i].segment.x1=lines2.at(i).x1;
  		segDescRight[i].segment.x2=lines2.at(i).x2;
  		segDescRight[i].segment.y1=lines2.at(i).y1;
  		segDescRight[i].segment.y2=lines2.at(i).y2;
  		segDescRight[i].segment.angle=lines2.at(i).angle;
  		segDescRight[i].segment.label=i;
  		segDescRight[i].lineLength=sqrt((lines2.at(i).x2-lines2.at(i).x1)*(lines2.at(i).x2-lines2.at(i).x1)+(lines2.at(i).y2-lines2.at(i).y1)*(lines2.at(i).y2-lines2.at(i).y1));
  		segDescRight[i].desVec=desc.calcLBD(lines2.at(i));
  		cout<<"LineID"<<segDescRight[i].segment.label<<endl;
  		for(int j=0;j<segDescRight[i].desVec.size();j++){
  			cout<<segDescRight[i].desVec[j]<<" ";
  		}
  		cout<<endl;
  	}
	BuildMat(segDescLeft,segDescRight);
	vector<unsigned int> matchResult;
	MatchingResult(segDescLeft,segDescRight,matchResult);
	for(int i=0;i<matchResult.size();i++){
		cout<<matchResult[i]<<" ";
	}
	int* r1=new int[matchResult.size()/2];
	int* g1=new int[matchResult.size()/2];
	int* b1=new int[matchResult.size()/2];
	for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++)
    {
        r1[pair] =  int(rand() % 256);
        g1[pair] =  int(rand() % 256);
        b1[pair] = 255 - r1[pair];
        double ww1 = 0.2 * (rand() % 5);
        double ww2 = 1 - ww1;
        char buf[10];
        sprintf(buf, "%d ", pair);
        int lineIDLeft = matchResult[2 * pair];
        int lineIDRight = matchResult[2 * pair + 1];
        Point startPoint = Point(int(segDescLeft[lineIDLeft].segment.x1), int(segDescLeft[lineIDLeft].segment.y1));
        Point endPoint = Point(int(segDescLeft[lineIDLeft].segment.x2), int(segDescLeft[lineIDLeft].segment.y2));
        Point midPoint=Point((int(segDescLeft[lineIDLeft].segment.x1)+int(segDescLeft[lineIDLeft].segment.x2))/2,(int(segDescLeft[lineIDLeft].segment.y1)+int(segDescLeft[lineIDLeft].segment.y2))/2);
        line(imgLeft, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, LINE_AA, 0);
        string textChar;
        stringstream ss;
        ss<<pair+1;
        ss>>textChar;
        putText(imgLeft,textChar,midPoint,FONT_HERSHEY_PLAIN,2,CV_RGB(r1[pair], g1[pair], b1[pair]),3);
        startPoint = Point(int(segDescRight[lineIDRight].segment.x1), int(segDescRight[lineIDRight].segment.y1));
        endPoint = Point(int(segDescRight[lineIDRight].segment.x2), int(segDescRight[lineIDRight].segment.y2));
        line(imgRight, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, LINE_AA, 0);
    }
	Mat ResultImage1 =Mat(Size(imgLeft.cols * 2, imgLeft.rows), imgLeft.type(), 3);
	Mat ResultImage2 =Mat(Size(imgLeft.cols * 2, imgLeft.rows), imgLeft.type(), 3);
	Mat ResultImage = Mat(Size(imgLeft.cols * 2, imgLeft.rows), imgLeft.type(), 3);
	Mat roi = ResultImage1(Rect(0, 0, imgLeft.cols, imgLeft.rows));
    resize(imgLeft, roi, roi.size(), 0, 0, 0);
	Mat roi2 = ResultImage1(Rect(imgLeft.cols, 0, imgLeft.cols, imgLeft.rows));
    resize(imgRight, roi2, roi2.size(), 0, 0, 0);
    ResultImage1.copyTo(ResultImage2);
    for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++)
    {
        int lineIDLeft = matchResult[2 * pair];
        int lineIDRight = matchResult[2 * pair + 1];
        Point startPoint = Point(int(segDescLeft[lineIDLeft].segment.x1), int(segDescLeft[lineIDLeft].segment.y1));
        Point endPoint = Point(int(segDescRight[lineIDRight].segment.x1 + imgLeft.cols), int(segDescRight[lineIDRight].segment.y1));
        line(ResultImage2, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 1, LINE_AA, 0);
    }
    addWeighted(ResultImage1, 0.5, ResultImage2, 0.5, 0.0, ResultImage, -1);
    for(int i=0;i<matchResult.size()/2;i++){
        cout<<"Line Pair"<<i+1<<endl;
        cout<<segDescLeft[matchResult[2*i]].segment.x1<<" "<<segDescLeft[matchResult[2*i]].segment.y1<<" "<<segDescLeft[matchResult[2*i]].segment.x2<<" "<<segDescLeft[matchResult[2*i]].segment.y2<<endl;
        cout<<segDescRight[matchResult[2*i+1]].segment.x1<<" "<<segDescRight[matchResult[2*i+1]].segment.y1<<" "<<segDescRight[matchResult[2*i+1]].segment.x2<<" "<<segDescRight[matchResult[2*i+1]].segment.y2<<endl;
        normalVector norm_v=getLineDirectionVector(segDescLeft[matchResult[2*i]].segment,segDescRight[matchResult[2*i+1]].segment,ResultImage);
        cout<<norm_v.nx<<" "<<norm_v.ny<<" "<<norm_v.nz<<endl;
    }
    imshow("matching", ResultImage);
    imshow("result1",imgLeftResult);
    imshow("result2",imgRightResult);
    waitKey(0);
    imwrite(argv[3],ResultImage);
    stereo s("../image/left1.png","../image/right1.png");*/
    /*vector<Point2f>featurePoints;
    featurePoints.resize(2*lines1.size());
    for(int i=0;i<lines1.size();i++){
        featurePoints[2*i].x=lines1.at(i).x1;
        featurePoints[2*i].y=lines1.at(i).y1;
        featurePoints[2*i+1].x=lines1.at(i).x2;
        featurePoints[2*i+1].y=lines1.at(i).y2;
    }
    vector<Point2f>matchPoints;
    vector<unsigned char>matchingStatus;
    vector<float>matchingErr;
    calcOpticalFlowPyrLK(im1,im2,featurePoints,matchPoints,matchingStatus,matchingErr);
    Mat im2Color=imread(argv[2]);
    for(int i=0;i<matchPoints.size();i++){
        line(im2Color,Point(matchPoints[2*i].x,matchPoints[2*i].y),Point(matchPoints[2*i+1].x,matchPoints[2*i+1].y),CV_RGB(0,0,255),1,LINE_AA,0);
    }
    imshow("Flow",im2Color);
    waitKey(0);
    Mat img1=imread(argv[1]);
    Mat img2=imread(argv[2]);
    Mat OrigImg1=imread(argv[1]);
    Mat OrigImg2=imread(argv[2]);
    vector<Point2f>intersectionLeft;
    vector<Point2f>intersectionRight;
    vector<int>linePairLeft;
    vector<int>linePairRight;
    vector< vector<float> >siftDescLeft;
    vector< vector<float> >siftDescRight;
    intersectionLeft.clear();
    intersectionRight.clear();
    linePairLeft.clear();
    linePairRight.clear();
    siftDescLeft.clear();
    siftDescRight.clear();
    for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++){
        int lineIDLeft1=matchResult[2*pair];
        int lineIDRight1=matchResult[2*pair+1];
        for(unsigned int pair2=pair+1;pair2<matchResult.size()/2;pair2++){
            int lineIDLeft2=matchResult[2*pair2];
            int lineIDRight2=matchResult[2*pair2+1];
            linePairLeft.push_back(lineIDLeft1);
            linePairLeft.push_back(lineIDLeft2);
            linePairRight.push_back(lineIDRight1);
            linePairRight.push_back(lineIDRight2);
            intersectionLeft.push_back(getIntersection(segDescLeft[lineIDLeft1].segment,segDescLeft[lineIDLeft2].segment));
            intersectionRight.push_back(getIntersection(segDescRight[lineIDRight1].segment,segDescRight[lineIDRight2].segment));
            int r =  int(rand() % 256);
            int g =  int(rand() % 256);
            int b = 255 - r1[pair];
            circle(img1,getIntersection(segDescLeft[lineIDLeft1].segment,segDescLeft[lineIDLeft2].segment),3,CV_RGB(r,g,b));
            circle(img2,getIntersection(segDescRight[lineIDRight1].segment,segDescRight[lineIDRight2].segment),3,CV_RGB(r,g,b));
        }
    }
    imshow("intersection1",img1);
    imshow("intersection2",img2);
    waitKey(0);*/
    return 0;
}