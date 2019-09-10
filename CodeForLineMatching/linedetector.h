#ifndef LINEDETECTOR_H_
#define LINEDETECTOR_H_

#include "system.h"

class LineDetector
{
public:
  LineDetector() {
    threshold_dist=1.5;
    threshold_length=20;
    init_label=0;
  }
  ~LineDetector(){};
	template<class tType>
    void incidentPoint( tType * pt, Mat & l );
	void mergeLines(SEGMENT * Seg1, SEGMENT * Seg2, SEGMENT * SegMerged);
  bool getPointChain( const Mat & img, const Point pt, Point * chained_pt, int
      & direction, int step );
  // bool getPointChain( const Mat & img, const Mat & scales, const Point pt, Point * chained_pt, int
  // & direction, int step );
  double distPointLine( const Mat & p, Mat & l );
	bool mergeSegments( SEGMENT * seg1, SEGMENT * seg2, SEGMENT * seg_merged );
	void extractSegments( vector<Point2i> * points, vector<SEGMENT> * segments );
	void lineDetection( Mat & src, vector<SEGMENT> & segments_all, bool merge = true );
	void pointInboardTest(Mat & src, Point2i * pt);
  void getAngle(SEGMENT *seg);
	void additionalOperationsOnSegments(Mat & src, SEGMENT * seg);
  void drawArrow( Mat& mat, const SEGMENT* seg, Scalar bgr=Scalar(0,255,0),
      int thickness=1, bool directed=true);

private:
	int init_label, imagewidth, imageheight, threshold_length;
  float threshold_dist;
};

struct segDesc{
	SEGMENT segment;
	float lineLength;
	vector<float>desVec;
};
struct matchNode{
	unsigned int leftLineID;//the index of line in the left image
	unsigned int rightLineID;//the index of line in the right image
};
typedef  std::vector<matchNode> Nodes_list;

struct CompareL {
    bool operator() (const double& lhs, const double& rhs) const
    {return lhs>rhs;}
};
typedef  std::multimap<double,unsigned int,CompareL> EigenMAP;



class descriptor{
  private:
  int numOfBand;
  int widthOfBand;
  vector<float> GaussianCoef_Global,GaussianCoef_Local,desc;
  Mat dxImg,dyImg;
  void setGlobalGaussianCoef(int n,int w);
  void setLocalGaussianCoef(int w);
  public:
  descriptor(int numberOfBands_,int widthOfBand_);
  void getGradientMap(string filename);
  void debugShow();
  vector<float> calcLBD(SEGMENT& seg);
};

void BuildMat(vector<segDesc> linesInLeft, vector<segDesc> linesInRight);
void MatchingResult(vector<segDesc> linesInLeft, vector<segDesc> linesInRight, vector<unsigned int> &matchResult);
#endif
