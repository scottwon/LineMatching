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
using namespace cv;//dependencies: OpenCV 4.0, contrib, modules
static const double pixelZ=438.4;
static const double trans=4.5;//intrinsic parameters of the stereo camera
static const double parallelBound=0.90;
static const double distBound=4.0;
static const double coplanarBound=0.15;//threshold values for coplanar analysis
GLfloat rtri=0.0;//for OpenGL automation
ofstream out;
Point2f getIntersection(SEGMENT& seg1,SEGMENT& seg2){//to find out the intersection of two line segments
    double x=(seg1.x2-seg1.x1)*(seg2.x1*seg2.y2-seg2.x2*seg2.y1)-(seg2.x2-seg2.x1)*(seg1.x1*seg1.y2-seg1.y1*seg1.x2);
    double y=(seg1.y1-seg1.y2)*(seg2.x1*seg2.y2-seg2.y1*seg2.x2)-(seg2.y1-seg2.y2)*(seg1.x1*seg1.y2-seg1.y1*seg1.x2);
    double r=(seg2.x2-seg2.x1)*(seg1.y1-seg1.y2)-(seg2.y1-seg2.y2)*(seg1.x2-seg1.x1);
    x=x/r;
    y=(-1)*y/r;
    return Point2f(x,y);
}
bool checkIntersection(SEGMENT& seg1,SEGMENT& seg2){//to determine whether two line segments intersect or not
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

struct homogeneousPoint{//the point structure in homogeneous coordination
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

struct normalVector{//the normal vector
    double nx,ny,nz;
    normalVector(){
        nx=0;ny=0;nz=0;
    }
    normalVector(double px,double py,double pz){
        nx=px;ny=py;nz=pz;
    }
    bool normalize(){//to normalize the vector to unit length, return false if failed
        double norm=sqrt(nx*nx+ny*ny+nz*nz);
        if(norm==0)return false;
        nx/=norm;
        ny/=norm;
        nz/=norm;
        return false;
    }
};

normalVector crossProduct(const homogeneousPoint &p1,const homogeneousPoint &p2){//to implement the cross product of two homogeneous points
    normalVector v(p1.y/p1.w*p2.z/p2.w-p2.y/p2.w*p1.z/p1.w,p1.z/p1.w*p2.x/p2.w-p2.z/p2.w*p1.x/p1.w,p1.x/p1.w*p2.y/p2.w-p1.y/p1.w*p2.x/p2.w);
    v.normalize();
    return v;
}

normalVector crossProduct(const normalVector &v1,const normalVector &v2){//to calculate the cross product of two (normal) vectors
    normalVector v(v1.ny*v2.nz-v1.nz*v2.ny,v1.nz*v2.nx-v1.nx*v2.nz,v1.nx*v2.ny-v2.nx*v1.ny);
    v.normalize();
    return v;
}

struct motionTranslation{//to implement the structure "Translation"
    double tx,ty,tz;
    motionTranslation(){
        tx=0;ty=0;tz=0;
    }
    motionTranslation(double px,double py,double pz){
        tx=px;ty=py;tz=pz;
    }
};

homogeneousPoint translation(const homogeneousPoint& p1,const motionTranslation& m){//to implement the translation of homogeneous points
    return homogeneousPoint(p1.x/p1.w+m.tx,p1.y/p1.w+m.ty,p1.z/p1.w+m.tz);
}

normalVector getNormalVector(const SEGMENT& seg,const Mat& img){//v1(x1,y1,1), v2(x2,y2,1), calculate the cross product of v1 and v2
    return crossProduct(homogeneousPoint(seg.x1-(img.cols-1)/2,(img.rows-1)/2-seg.y1),homogeneousPoint(seg.x2-(img.cols-1)/2,(img.rows-1)/2-seg.y2));//note: the center of the image is (0,0,1)
}

normalVector getLineDirectionVector(const SEGMENT& seg1,const SEGMENT& seg2,const Mat& img){//v1(x1,y1,1), v2(x2,y2,1), v3(x3,y3,1), v4(x4,y4,1), O(0,0,0), assume the intersection of plane Ov1v2 and plane Ov3v4 is l, find out the unit length directional vector of l
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

double getAngle(const line3D& l1, const line3D& l2){//return the cosine value of the angle between two 3D lines
    return l1.direction[0]*l2.direction[0]+l1.direction[1]*l2.direction[1]+l1.direction[2]*l2.direction[2];
}

double getDist(const line3D& l1,const line3D& l2){//to calculate the distance between two 3D lines
    normalVector n1(l1.direction[0],l1.direction[1],l1.direction[2]);
    normalVector n2(l2.direction[0],l2.direction[1],l2.direction[2]);
    normalVector n=crossProduct(n1,n2);//n is a unit vector that is perpendicular to both l1 and l2
    normalVector v(l2.leftEndPoint1[0]-l1.leftEndPoint1[0],l2.leftEndPoint1[1]-l1.leftEndPoint1[1],l2.leftEndPoint1[2]-l1.leftEndPoint1[2]);//v is the vector from l1's left endpoint to l2's left endpoint
    return abs(v.nx*n.nx+v.ny*n.ny+v.nz*n.nz);//n dot v is the distance from l1 to l2
}

Vec3f getNormalVector(const line3D& l1,const line3D& l2){//to get the normal vector that is perpendicular to both l1 and l2
    Vec3f tmp;
    double len;
    tmp[0]=l1.direction[1]*l2.direction[2]-l1.direction[2]*l2.direction[1];
    tmp[1]=l1.direction[2]*l2.direction[0]-l1.direction[0]*l2.direction[2];
    tmp[2]=l1.direction[0]*l2.direction[1]-l1.direction[1]*l2.direction[0];
    len=sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
    tmp[0]=tmp[0]/len;
    tmp[1]=tmp[1]/len;
    tmp[2]=tmp[2]/len;
    return tmp;//tmp is unitary
}

double getAngle(const line3D& l,const Vec3f& nv){//to get the cosine value of the angle between l and nv
    double tmp=l.direction[0]*nv[0]+l.direction[1]*nv[1]+l.direction[2]*nv[2];
    double len=sqrt(l.direction[0]*l.direction[0]+l.direction[1]*l.direction[1]+l.direction[2]*l.direction[2]);
    return tmp/len;
}

void cvFitPlane(const Mat& points, float* plane){//to find out the float* plane that can fit the Mat& points best. The least square criteria is applied. Singular Value Decomposition is used in the calculation process
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
    Mat A,W,U,V;//note: the interface of SVD varies for different editions of OpenCVs
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

void fitPlane(vector<line3D>& coplanarLines,float* plane){//to find out the best-fit plane for a bundle of co-planar lines
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

bool fitLine(const line3D& l,float* plane,double bound=distBound){//to determine whether the line l is close enough to the plane. distBound is the threshold value
    if(abs(l.leftEndPoint1[0]*plane[0]+l.leftEndPoint1[1]*plane[1]*l.leftEndPoint1[2]*plane[2]-plane[3])>distBound)return false;
    if(abs(l.leftEndPoint2[0]*plane[0]+l.leftEndPoint2[1]*plane[1]+l.leftEndPoint2[2]*plane[2]-plane[3])>distBound)return false;
    return true;
}

typedef vector<line3D> PlaneRecon;
typedef vector<PlaneRecon> structure3D;

class stereo{
    public:
        static double focalLength;//intrinsic parameter of the stereo camera
        static Vec3f translation;
        Mat imgLeft,imgRight,imgLeftGray,imgRightGray;//store the stereo images and the grayscale images
        int imageWidth,imageHeight;//store the size of the stereo images
        LineDetector ld;//Linedetection is implemented by linedetector.h, linedetector.cpp. The main contributors of this part is a research team from Korea
        vector<SEGMENT>linesLeft,linesRight;//store the detected line segments in stereo images
        vector<unsigned int> matchResult;//store the result of line matching
        vector<segDesc> segDescLeft,segDescRight;//store the LBD descriptors of each line segments
        vector<line3D>lineRecon;//calculate the 3D line info from the matching line pair
        vector<normalVector>leftPlanes,rightPlanes;
        vector<double>colorR,colorG,colorB;//a random table of colors, used for drawing in OpenGL
        vector<vector<lineRelations> >lineRelMat;//to determine the geometric relations among the reconstructed 3D lines
        vector<vector<Point2f> >featuresLeft,featuresRight;//to store the endpoints and mid-point of the left segment and the right segment
        vector<vector<Mat> >homographies;//to assume that line[i] and line[j] are co-planar and calculate the planar homographies
        structure3D planeStructure;
        vector<vector<float> >PlaneFitEqu;
        vector<Mat>PlaneHomographies;
        vector<Mat>constraintH;
//      vector<vector<Mat> >transImg;
    public:
        stereo(){
        }
        void set(const string& filenameLeft,const string& filenameRight){//line detection, line matching, drawing, the calculation of the reconstructed 3D lines, the calculation of planar homographies
            imgLeft=imread(filenameLeft);
            imgRight=imread(filenameRight);
            cvtColor(imgLeft,imgLeftGray,COLOR_RGB2GRAY);
            cvtColor(imgRight,imgRightGray,COLOR_RGB2GRAY);
            imageWidth=imgLeft.cols;
            imageHeight=imgLeft.rows;
            linesLeft.clear();
            linesRight.clear();
            ld.lineDetection(imgLeftGray,linesLeft);
            ld.lineDetection(imgRightGray,linesRight);//detect the lines in stereo images
            descriptor desc(9,7);//initialize the parameters for LBD descriptor
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
  		        segDescLeft[i].desVec=desc.calcLBD(linesLeft.at(i));//calculate the LBD descriptors for each detected lines in the left image
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
  		        segDescRight[i].desVec=desc.calcLBD(linesRight.at(i));//calculate the LBD descriptors for each detected lines in the right image
  	        }
	        BuildMat(segDescLeft,segDescRight);
            matchResult.clear();
	        MatchingResult(segDescLeft,segDescRight,matchResult);//store the line matching result
            leftPlanes.clear();
            rightPlanes.clear();
            lineRecon.clear();
            srand((int)time(0));
            colorR.clear();
            colorG.clear();
            colorB.clear();
            colorR.resize(matchResult.size()/2);
            colorG.resize(matchResult.size()/2);
            colorB.resize(matchResult.size()/2);
            for(int i=0;i<matchResult.size()/2;i++){//assign colors randomly to each line pair
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
                sss>>iid;//iid is the ID number of the line segment
                line(img,Point(segDescLeft[matchResult[2*i]].segment.x1,segDescLeft[matchResult[2*i]].segment.y1),Point(segDescLeft[matchResult[2*i]].segment.x2,segDescLeft[matchResult[2*i]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
                putText(img,iid,Point((segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2,(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
                line(img2,Point(segDescRight[matchResult[2*i+1]].segment.x1,segDescRight[matchResult[2*i+1]].segment.y1),Point(segDescRight[matchResult[2*i+1]].segment.x2,segDescRight[matchResult[2*i+1]].segment.y2),Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3,LINE_AA);
                putText(img2,iid,Point((segDescRight[matchResult[2*i+1]].segment.x1+segDescRight[matchResult[2*i+1]].segment.x2)/2,(segDescRight[matchResult[2*i+1]].segment.y1+segDescRight[matchResult[2*i+1]].segment.y2)/2),FONT_HERSHEY_PLAIN,2,Scalar(colorB[i]*255,colorG[i]*255,colorR[i]*255),3);
                //draw matched line pairs in both the left image and the right image
            }
            imshow("Left View",img);
            imshow("Right View",img2);
            imwrite("outleft.png",img);
            imwrite("outright.png",img2);//output the result of detection and matching
            waitKey(0);//press any key to continue
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
                //to calculate the reconstructed 3D lines
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
                //store the geometric information of the 3D lines in files
                out<<"Line ID"<<i+1<<":"<<endl;
                out<<"Left:"<<endl;
                out<<"Start Point:"<<l.leftEndPoint1[0]<<" "<<l.leftEndPoint1[1]<<" "<<l.leftEndPoint1[2]<<endl;
                out<<"End Point:"<<l.leftEndPoint2[0]<<" "<<l.leftEndPoint2[1]<<" "<<l.leftEndPoint2[2]<<endl;
                out<<"Mid Point:"<<l.leftMidPoint[0]<<" "<<l.leftMidPoint[1]<<" "<<l.leftMidPoint[2]<<endl;
                out<<"Left normal Vector"<<n1.nx<<" "<<n1.ny<<" "<<n1.nz<<endl;
                out<<"Right:"<<endl;
                out<<"Start Point:"<<l.rightEndPoint1[0]<<" "<<l.rightEndPoint1[1]<<" "<<l.rightEndPoint1[2]<<endl;
                out<<"End Point:"<<l.rightEndPoint2[0]<<" "<<l.rightEndPoint2[1]<<" "<<l.rightEndPoint2[2]<<endl;
                out<<"Mid Point:"<<l.rightMidPoint[0]<<" "<<l.rightMidPoint[1]<<" "<<l.rightMidPoint[2]<<endl;
                out<<"Right normal Vector"<<n2.nx<<" "<<n2.ny<<" "<<n2.nz<<endl;                
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
                tmp.x=segDescLeft[matchResult[2*i]].segment.x1;//endpoint 1 of the left line segment
                tmp.y=segDescLeft[matchResult[2*i]].segment.y1;
                featuresLeft[i].push_back(tmp);
                tmp.x=segDescLeft[matchResult[2*i]].segment.x2;//endpoint 2 of the left line segment
                tmp.y=segDescLeft[matchResult[2*i]].segment.y2;
                featuresLeft[i].push_back(tmp);
                tmp.x=(segDescLeft[matchResult[2*i]].segment.x1+segDescLeft[matchResult[2*i]].segment.x2)/2;//mid-point of the left line segment
                tmp.y=(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2;
                featuresLeft[i].push_back(tmp);
                //store endpoint 1 endpoint 2 mid-point of the left line segment into featuresLeft
                tmp.x=(segDescLeft[matchResult[2*i]].segment.y1-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y1;
                featuresRight[i].push_back(tmp);
                tmp.x=(segDescLeft[matchResult[2*i]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=segDescLeft[matchResult[2*i]].segment.y2;
                featuresRight[i].push_back(tmp);
                tmp.x=((segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2-segDescRight[matchResult[2*i+1]].segment.y2)*(segDescRight[matchResult[2*i+1]].segment.x2-segDescRight[matchResult[2*i+1]].segment.x1)/(segDescRight[matchResult[2*i+1]].segment.y2-segDescRight[matchResult[2*i+1]].segment.y1)+segDescRight[matchResult[2*i+1]].segment.x2;
                tmp.y=(segDescLeft[matchResult[2*i]].segment.y1+segDescLeft[matchResult[2*i]].segment.y2)/2;
                featuresRight[i].push_back(tmp);
                //store the matched endpoint 1 endpoint 2 mid-point of the right line segment into featuresRight
            }
            out<<"Line Relations:"<<endl;
//            Mat pointMatch=imread("../image/right1.png");
            for(int i=0;i<lineRelMat.size();i++){
                for(int j=i+1;j<lineRelMat.size();j++){
                //    out<<"i "<<lineRecon[i].direction<<endl;
                //    out<<"j "<<lineRecon[j].direction<<endl;
                    lineRelMat[i][j].angle=getAngle(lineRecon[i],lineRecon[j]);//calculate the angle and dist between reconstructed 3D lines
                    lineRelMat[j][i].angle=lineRelMat[i][j].angle;
                    out<<i+1<<" "<<j+1<<" "<<lineRelMat[i][j].angle<<endl;
                    lineRelMat[i][j].dist=getDist(lineRecon[i],lineRecon[j]);
                    lineRelMat[j][i].dist=lineRelMat[i][j].dist;
                    out<<lineRelMat[i][j].dist<<endl;
                    //line(pointMatch,Point(PointsRight[0].x,PointsRight[0].y),Point(PointsRight[1].x,PointsRight[1].y),Scalar(255,0,0),1);
                    //line(pointMatch,Point(PointsRight[2].x,PointsRight[2].y),Point(PointsRight[3].x,PointsRight[3].y),Scalar(255,0,0),1);
                    vector<Point2f>PointsLeft,PointsRight;
                    PointsLeft.clear();
                    PointsRight.clear();
                    PointsLeft.insert(PointsLeft.end(),featuresLeft[i].begin(),featuresLeft[i].end());
                    PointsLeft.insert(PointsLeft.end(),featuresLeft[j].begin(),featuresLeft[j].end());
                    PointsRight.insert(PointsRight.end(),featuresRight[i].begin(),featuresRight[i].end());
                    PointsRight.insert(PointsRight.end(),featuresRight[j].begin(),featuresRight[j].end());
                    homographies[i][j]=findHomography(PointsLeft,PointsRight,RANSAC);//to find out the planar homographies
                    out<<homographies[i][j]<<endl;
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
            waitKey(0);//press any key to continue
            out.close();  
        }
        void getPlanes(){//co-planar analysis
            planeStructure.clear();
            PlaneFitEqu.clear();
            bool* checked=new bool[homographies.size()*homographies.size()];
            memset(checked,0,sizeof(bool)*homographies.size()*homographies.size());
            for(int i=0;i<homographies.size();i++){
                for(int j=i+1;j<homographies.size();j++){
                    if(checked[i*homographies.size()+j]==true)continue;
                    if(abs(lineRelMat[i][j].angle)<parallelBound && lineRelMat[i][j].dist>distBound){//skip the skew lines
                        checked[i*homographies.size()+j]=true;
                        checked[j*homographies.size()+i]=true;
                        continue;
                    }
                    if(abs(lineRelMat[i][j].angle)>parallelBound)continue;//skip the parallel lines at first
                    vector<line3D>tmp;//to deal with the intersecting lines
                    tmp.clear();//tmp is a coplanar hypothesis
                    tmp.push_back(lineRecon[i]);//the intersecting lines i,j are in the coplanar hypothesis initially
                    tmp.push_back(lineRecon[j]);
                    checked[i*homographies.size()+j]=true;//line pair i,j are checked
                    checked[j*homographies.size()+i]=true;
                    Vec3f nv=getNormalVector(lineRecon[i],lineRecon[j]);
                    for(int k=0;k<homographies.size();k++){//to traverse the 3D lines
                        bool flag=true;
                        bool cross=false;
                        if(k==i||k==j)continue;
                        if(abs(getAngle(lineRecon[k],nv))>coplanarBound)continue;//skip if line k is not parallel to the plane hypothesis
                        for(int l=0;l<tmp.size();l++){//check the geometric relations of k and any line l in the coplanar hypothesis
                            if(lineRelMat[k][tmp[l].ID].dist>distBound && abs(lineRelMat[k][tmp[l].ID].angle)<parallelBound){//if k,l are skew lines, skip
                                flag=false;
                                break;
                            }
                            if(abs(lineRelMat[k][tmp[l].ID].angle)<parallelBound && lineRelMat[k][tmp[l].ID].dist<distBound)cross=true;//if k,l are intersecting lines, set cross to be true
                        }
                        //flag==true means that k is parallel or intersecting to all the lines l in the coplanar hypothesis
                        //cross==true means that k is intersecting to at least one line l in the coplanar hypothesis
                        if(flag&&cross){//at such a condition, the line k should be merged into the coplanar hypothesis
                            for(int l=0;l<tmp.size();l++){
                                checked[k*homographies.size()+tmp[l].ID]=true;
                                checked[(tmp[l].ID)*homographies.size()+k]=true;
                            }
                            tmp.push_back(lineRecon[k]);
                        }
                    }
                    //to see whether the new coplanar hypothesis can be merged with the existing ones
                    vector<int> repeatedIDs;
                    int mergeIndex=-1;
                    for(int k1=0;k1<planeStructure.size();k1++){//to check all the existing coplanar hypothesis
                        if(mergeIndex>-1)break;
                        repeatedIDs.clear();
                        for(int k2=0;k2<tmp.size();k2++){
                            if(mergeIndex>-1)break;
                            for(int k3=0;k3<planeStructure[k1].size();k3++){
                                if(tmp[k2].ID==planeStructure[k1][k3].ID){//if there is a line ID exist in both coplanar hypothesis
                                    if(repeatedIDs.size()==0){//if there is only one repeated ID, it doesn't mean that the coplanar hypothesis should be merged
                                        repeatedIDs.push_back(tmp[k2].ID);
                                        break;
                                    }
                                    else{//if two repeated IDs are parallel or intersecting
                                        if((lineRelMat[repeatedIDs[0]][tmp[k2].ID].angle<parallelBound && lineRelMat[repeatedIDs[0]][tmp[k2].ID].dist<distBound)||(lineRelMat[repeatedIDs[0]][tmp[k2].ID].angle>parallelBound && lineRelMat[repeatedIDs[0]][tmp[k2].ID].dist>distBound)){
                                            repeatedIDs.push_back(tmp[k2].ID);
                                        }
                                        if(repeatedIDs.size()>1)mergeIndex=k1;//the coplanar hypothesis should be merged
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if(mergeIndex>-1){//merge and re-calculate the plane equations
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
                    else{//add the planar hypothesis to planeStructure and calculate the estimated plane equations
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
                    if(checked[i*homographies.size()+j]==false){//find the parallel lines
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
                            //filter out the pseudo-planes
                            //it seems that the code here need to be improved. It is such a pity that the latest bug-free version of my code is somehow lost here
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
                    if(checked[i*homographies.size()+j]==false){//the remaining parallel lines are likely to form planes
                    //form a coplanar hypothesis from two parallel lines
                        vector<line3D>linePairs;
                        linePairs.clear();
                        linePairs.push_back(lineRecon[i]);
                        linePairs.push_back(lineRecon[j]);
                        float fitEqu[4];
                        fitPlane(linePairs,fitEqu);
                        bool checkMerge=false;//check if the coplanar hypothesis should be merged with the existing ones
                        for(int k=0;k<parallelPairs.size();k++){
                            if(checkMerge)break;
                            bool coincidenceI=false;
                            bool coincidenceJ=false;
                            for(int l=0;l<parallelPairs[k].size();l++){
                                if(parallelPairs[k][l].ID==i && abs(parallelFit[k][0]*fitEqu[0]+parallelFit[k][1]*fitEqu[1]+parallelFit[k][2]*fitEqu[2])>parallelBound)coincidenceI=true;
                                if(parallelPairs[k][l].ID==j && abs(parallelFit[k][0]*fitEqu[0]+parallelFit[k][1]*fitEqu[1]+parallelFit[k][2]*fitEqu[2])>parallelBound)coincidenceJ=true;
                                if(coincidenceI && coincidenceJ){//if there are two repeated IDs, then the coplanar hypothesis should be merged
                                    checkMerge=true;
                                    break;
                                }
                            }
                            if(coincidenceI && ! coincidenceJ){//merge and update the equations
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
            /*planeStructure.clear();
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
                    }
                    else{
                        planeStructure.push_back(tmp);
                    }
                }
            }*/
            /*for(int i=0;i<homographies.size();i++){
                for(int j=i+1;j<homographies.size();j++){
                    if(checked[i*homographies.size()+j]==false){
                        cout<<i+1<<" "<<j+1<<endl;
                        Mat im=imread("../image/left1.png");
                        line(im,Point(segDescLeft[matchResult[2*i]].segment.x1,segDescLeft[matchResult[2*i]].segment.y1),Point(segDescLeft[matchResult[2*i]].segment.x2,segDescLeft[matchResult[2*i]].segment.y2),Scalar(255,0,0),3);
                        line(im,Point(segDescLeft[matchResult[2*j]].segment.x1,segDescLeft[matchResult[2*j]].segment.y1),Point(segDescLeft[matchResult[2*j]].segment.x2,segDescLeft[matchResult[2*j]].segment.y2),Scalar(255,0,0),3);
                        imshow("Parallel",im);
                        waitKey(0);
                    }
                }
            }*/
            delete[] checked;
            cout<<planeStructure.size()<<endl;
            for(int i=0;i<planeStructure.size();i++){//output the planes
                Mat tmp;
                tmp=imread("../image/left1.png");
                for(int j=0;j<planeStructure[i].size();j++){
                    cout<<planeStructure[i][j].ID+1<<" ";
                    line(tmp,Point(segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.x1,segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.y1),Point(segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.x2,segDescLeft[matchResult[2*planeStructure[i][j].ID]].segment.y2),Scalar(255,0,0),3);
                }
                cout<<endl;
                imshow("",tmp);
                stringstream fn;
                string ttl;
                fn<<"Plane"<<i+1<<".png";
                fn>>ttl;
                imwrite(ttl,tmp);
                waitKey(0);
            }
        }
};


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
void display(void){//draw the 3D lines in OpenGL
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

int main(int argc, char **argv){
    out.open("info.txt");
    s.set("../image/left1.png","../image/right1.png");//line detection and matching
    s.getPlanes();//coplanar analysis
    glutInit(&argc, argv);//OpenGL drawing
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