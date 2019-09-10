#include "linedetector.h"
#include <string>
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

using namespace std;
using namespace cv;

#define DescriptorDifThreshold 0.35
#define LengthDifThreshold 4
#define RelativeAngleDifferenceThreshold 0.7854
#define Inf 1e10
#define ProjectionRationDifThreshold 0.06
#define IntersectionRationDifThreshold 0.06
#define WeightOfMeanEigenVec 0.1

Nodes_list nodesList_;
EigenMAP eigenMap_;
double minOfEigenVec_;

void descriptor::setGlobalGaussianCoef(int n,int w){
    GaussianCoef_Global.resize(n*w);
    float sigma=(n*w-1)/2;
    float invSigma2=-1.0/(2.0*sigma*sigma);
    for(int i=0;i<n*w;i++){
       float dist=i-sigma;
        GaussianCoef_Global[i]=exp(dist*dist*invSigma2);
    }
}
void descriptor::setLocalGaussianCoef(int w){
    GaussianCoef_Local.resize(w*3);
    float u=(w*3-1)/2;
    float sigma=(w*2+1)/2;
    float invSigma2=-1.0/(2.0*sigma*sigma);
    for(int i=0;i<w*3;i++){
        float dist=i-u;
        GaussianCoef_Local[i]=exp(dist*dist*invSigma2);
    }
}
descriptor::descriptor(int numberOfBands_=9,int widthOfBand_=7){
    numOfBand=numberOfBands_;
    widthOfBand=widthOfBand_;
    setGlobalGaussianCoef(numOfBand,widthOfBand);
    setLocalGaussianCoef(widthOfBand);
}
void descriptor::getGradientMap(string filename){
    Mat image=imread(filename);
        //error handling codes to be added
    Sobel( image, dxImg, CV_16SC1, 1, 0, 3);//16SC1 in original version
	Sobel( image, dyImg, CV_16SC1, 0, 1, 3);
}
void descriptor::debugShow(){
    imshow("dx",dxImg);
    imshow("dy",dyImg);
    waitKey(0);
}
vector<float> descriptor::calcLBD(SEGMENT& seg){
float *dLine = new float[2];
    float *dOrtho = new float[2];
    short descriptorSize = numOfBand * 8;
    float positiveGradientLineRowSum;
    float negativeGradientLineRowSum;
    float positiveGradientLine2RowSum;
    float negativeGradientLine2RowSum;
    float positiveGradientOrthoRowSum;
    float negativeGradientOrthoRowSum;
    float positiveGradientOrtho2RowSum;
    float negativeGradientOrtho2RowSum;
   	float *positiveGradientLineBandSum  = new float[numOfBand];
    float *negativeGradientLineBandSum  = new float[numOfBand];
    float *positiveGradientLine2BandSum = new float[numOfBand];
    float *negativeGradientLine2BandSum = new float[numOfBand];
    float *positiveGradientOrthoBandSum  = new float[numOfBand];
    float *negativeGradientOrthoBandSum  = new float[numOfBand];
    float *positiveGradientOrtho2BandSum = new float[numOfBand];
    float *negativeGradientOrtho2BandSum = new float[numOfBand];
    short numOfBitsBand = numOfBand*sizeof(float);
    short heightOfLineSupportRegion = widthOfBand*numOfBand;
    short lengthOfLineSupportRegion; 
    short halfHeight = (heightOfLineSupportRegion-1)/2;
    short halfWidth;
    short bandID;
    float coefInGaussion;
    float lineMiddlePointX, lineMiddlePointY;
    float sCorX, sCorY,sCorX0, sCorY0;
    short tempCor, xCor, yCor;//pixel coordinates in image plane
    short dx, dy;
    float gradientLine;//store the gradient projection of pixels in support region along dL vector
    float gradientOrtho;//store the gradient projection of pixels in support region along dO vector
    short imageWidth, imageHeight, realWidth;
    short *pdxImg, *pdyImg;
    float *desVec;
    short sameLineSize;
    //short octaveCount;
       //pSingleLine = &(keyLines[lineIDInScaleVec][lineIDInSameLine]);
	//	octaveCount = pSingleLine->octaveCount;
    pdxImg=dxImg.ptr<short>();
    pdyImg=dyImg.ptr<short>();
    realWidth=dxImg.cols;
	imageWidth=realWidth-1;
	imageHeight=dxImg.rows-1;
		
	memset(positiveGradientLineBandSum,  0, numOfBitsBand);
	memset(negativeGradientLineBandSum, 0, numOfBitsBand);
	memset(positiveGradientLine2BandSum,  0, numOfBitsBand);
	memset(negativeGradientLine2BandSum, 0, numOfBitsBand);
	memset(positiveGradientOrthoBandSum,  0, numOfBitsBand);
	memset(negativeGradientOrthoBandSum, 0, numOfBitsBand);
	memset(positiveGradientOrtho2BandSum,  0, numOfBitsBand);
	memset(negativeGradientOrtho2BandSum, 0, numOfBitsBand);

	lengthOfLineSupportRegion = round(sqrt((seg.x2-seg.x1)*(seg.x2-seg.x1)+(seg.y2-seg.y1)*(seg.y2-seg.y1)));
		//should be numOfPixels instead
	halfWidth   = (lengthOfLineSupportRegion-1)/2;
	lineMiddlePointX = 0.5 * (seg.x1 +  seg.x2);
	lineMiddlePointY = 0.5 * (seg.y1 +  seg.y2);
		/*1.rotate the local coordinate system to the line direction
			 *2.compute the gradient projection of pixels in line support region*/
	dLine[0] = cos(seg.angle);
	dLine[1] = sin(seg.angle);
	dOrtho[0] = -dLine[1];
	dOrtho[1] = dLine[0];
	sCorX0= -dLine[0]*halfWidth + dLine[1]*halfHeight + lineMiddlePointX;//hID =0; wID = 0;
	sCorY0= -dLine[1]*halfWidth - dLine[0]*halfHeight + lineMiddlePointY;
		//      BIAS::Matrix<float> gDLMat(heightOfLSP,lengthOfLSP);
	for(short hID = 0; hID <heightOfLineSupportRegion; hID++){
			//initialization
		sCorX = sCorX0;
		sCorY = sCorY0;

		positiveGradientLineRowSum = 0;
		negativeGradientLineRowSum = 0;
		positiveGradientOrthoRowSum = 0;
		negativeGradientOrthoRowSum = 0;

		for(short wID = 0; wID <lengthOfLineSupportRegion; wID++){
			tempCor = round(sCorX);
			xCor = (tempCor<0)?0:(tempCor>imageWidth)?imageWidth:tempCor;
			tempCor = round(sCorY);
			yCor = (tempCor<0)?0:(tempCor>imageHeight)?imageHeight:tempCor;
					/* To achieve rotation invariance, each simple gradient is rotated aligned with
					 * the line direction and clockwise orthogonal direction.*/
			dx = pdxImg[yCor*realWidth+xCor];
			dy = pdyImg[yCor*realWidth+xCor];
			gradientLine = dx * dLine[0] + dy * dLine[1];
			gradientOrtho = dx * dOrtho[0] + dy * dOrtho[1];
			if(gradientLine>0){
				positiveGradientLineRowSum  += gradientLine;
			}else{
				negativeGradientLineRowSum  -= gradientLine;
			}
			if(gradientOrtho>0){
				positiveGradientOrthoRowSum  += gradientOrtho;
			}else{
				negativeGradientOrthoRowSum  -= gradientOrtho;
			}
			sCorX +=dLine[0];
			sCorY +=dLine[1];
				//					gDLMat[hID][wID] = gDL;
		}
		sCorX0 -=dLine[1];
		sCorY0 +=dLine[0];
		coefInGaussion = GaussianCoef_Global[hID];
		positiveGradientLineRowSum = coefInGaussion * positiveGradientLineRowSum;
		negativeGradientLineRowSum = coefInGaussion * negativeGradientLineRowSum;
		positiveGradientLine2RowSum = positiveGradientLineRowSum * positiveGradientLineRowSum;
		negativeGradientLine2RowSum = negativeGradientLineRowSum * negativeGradientLineRowSum;
		positiveGradientOrthoRowSum = coefInGaussion * positiveGradientOrthoRowSum;
		negativeGradientOrthoRowSum = coefInGaussion * negativeGradientOrthoRowSum;
		positiveGradientOrtho2RowSum = positiveGradientOrthoRowSum * positiveGradientOrthoRowSum;
		negativeGradientOrtho2RowSum = negativeGradientOrthoRowSum * negativeGradientOrthoRowSum;
				//compute {g_dL |g_dL>0 }, {g_dL |g_dL<0 },
				//{g_dO |g_dO>0 }, {g_dO |g_dO<0 } of each band in the line support region
				//first, current row belong to current band;
		bandID = hID/widthOfBand;
		coefInGaussion = GaussianCoef_Local[hID%widthOfBand+widthOfBand];
		positiveGradientLineBandSum[bandID] +=  coefInGaussion * positiveGradientLineRowSum;
		negativeGradientLineBandSum[bandID] +=  coefInGaussion * negativeGradientLineRowSum;
		positiveGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientLine2RowSum;
		negativeGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientLine2RowSum;
		positiveGradientOrthoBandSum[bandID] +=  coefInGaussion * positiveGradientOrthoRowSum;
		negativeGradientOrthoBandSum[bandID] +=  coefInGaussion * negativeGradientOrthoRowSum;
		positiveGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientOrtho2RowSum;
		negativeGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientOrtho2RowSum;
				/* In order to reduce boundary effect along the line gradient direction,
				 * a row's gradient will contribute not only to its current band, but also
				 * to its nearest upper and down band with gaussCoefL_.*/
		bandID--;
		if(bandID>=0){//the band above the current band
			coefInGaussion = GaussianCoef_Local[hID%widthOfBand + 2*widthOfBand];
			positiveGradientLineBandSum[bandID] +=  coefInGaussion * positiveGradientLineRowSum;
			negativeGradientLineBandSum[bandID] +=  coefInGaussion * negativeGradientLineRowSum;
			positiveGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientLine2RowSum;
			negativeGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientLine2RowSum;
			positiveGradientOrthoBandSum[bandID] +=  coefInGaussion * positiveGradientOrthoRowSum;
			negativeGradientOrthoBandSum[bandID] +=  coefInGaussion * negativeGradientOrthoRowSum;
			positiveGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientOrtho2RowSum;
			negativeGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientOrtho2RowSum;
		}
		bandID = bandID+2;
		if(bandID<numOfBand){//the band below the current band
			coefInGaussion = GaussianCoef_Local[hID%widthOfBand];
			positiveGradientLineBandSum[bandID] +=  coefInGaussion * positiveGradientLineRowSum;
			negativeGradientLineBandSum[bandID] +=  coefInGaussion * negativeGradientLineRowSum;
			positiveGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientLine2RowSum;
			negativeGradientLine2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientLine2RowSum;
			positiveGradientOrthoBandSum[bandID] +=  coefInGaussion * positiveGradientOrthoRowSum;
			negativeGradientOrthoBandSum[bandID] +=  coefInGaussion * negativeGradientOrthoRowSum;
			positiveGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * positiveGradientOrtho2RowSum;
			negativeGradientOrtho2BandSum[bandID] +=  coefInGaussion * coefInGaussion * negativeGradientOrtho2RowSum;
		}
	}
			//			gDLMat.Save("gDLMat.txt");
			//			return 0;
			//construct line descriptor
	desc.resize(descriptorSize);
	desVec = desc.data();
	short desID;
		/*Note that the first and last bands only have (lengthOfLSP * widthOfBand_ * 2.0) pixels
			 * which are counted. */
	float invN2 = 1.0/(widthOfBand * 2.0);
	float invN3 = 1.0/(widthOfBand * 3.0);
	float invN, temp;
	for(bandID = 0; bandID<numOfBand; bandID++){
		if(bandID==0||bandID==numOfBand-1){	
            invN = invN2;
		}
        else{ 
            invN = invN3;
        }
		desID = bandID * 8;
		temp = positiveGradientLineBandSum[bandID] * invN;
		desVec[desID]   = temp;//mean value of pgdL;
		desVec[desID+4] = sqrt(positiveGradientLine2BandSum[bandID] * invN - temp*temp);//std value of pgdL;
		temp = negativeGradientLineBandSum[bandID] * invN;
		desVec[desID+1] = temp;//mean value of ngdL;
		desVec[desID+5] = sqrt(negativeGradientLine2BandSum[bandID] * invN - temp*temp);//std value of ngdL;

		temp = positiveGradientOrthoBandSum[bandID] * invN;
		desVec[desID+2] = temp;//mean value of pgdO;
		desVec[desID+6] = sqrt(positiveGradientOrtho2BandSum[bandID] * invN - temp*temp);//std value of pgdO;
		temp = negativeGradientOrthoBandSum[bandID] * invN;
		desVec[desID+3] = temp;//mean value of ngdO;
		desVec[desID+7] = sqrt(negativeGradientOrtho2BandSum[bandID] * invN - temp*temp);//std value of ngdO;
	}
		//normalize;
	float tempM, tempS;
	tempM = 0;
	tempS = 0;
	desVec = desc.data();
	for(short i=0; i<numOfBand; i++){
		tempM += (*desVec) * *(desVec++);//desVec[8*i+0] * desVec[8*i+0];
		tempM += (*desVec) * *(desVec++);//desVec[8*i+1] * desVec[8*i+1];
		tempM += (*desVec) * *(desVec++);//desVec[8*i+2] * desVec[8*i+2];
		tempM += (*desVec) * *(desVec++);//desVec[8*i+3] * desVec[8*i+3];
		tempS += (*desVec) * *(desVec++);//desVec[8*i+4] * desVec[8*i+4];
		tempS += (*desVec) * *(desVec++);//desVec[8*i+5] * desVec[8*i+5];
		tempS += (*desVec) * *(desVec++);//desVec[8*i+6] * desVec[8*i+6];
		tempS += (*desVec) * *(desVec++);//desVec[8*i+7] * desVec[8*i+7];
	}
	tempM = 1/sqrt(tempM);
	tempS = 1/sqrt(tempS);
	desVec = desc.data();
	for(short i=0; i<numOfBand; i++){
		(*desVec) = *(desVec++) * tempM;//desVec[8*i] =  desVec[8*i] * tempM;
		(*desVec) = *(desVec++) * tempM;//desVec[8*i+1] =  desVec[8*i+1] * tempM;
		(*desVec) = *(desVec++) * tempM;//desVec[8*i+2] =  desVec[8*i+2] * tempM;
		(*desVec) = *(desVec++) * tempM;//desVec[8*i+3] =  desVec[8*i+3] * tempM;
		(*desVec) = *(desVec++) * tempS;//desVec[8*i+4] =  desVec[8*i+4] * tempS;
		(*desVec) = *(desVec++) * tempS;//desVec[8*i+5] =  desVec[8*i+5] * tempS;
		(*desVec) = *(desVec++) * tempS;//desVec[8*i+6] =  desVec[8*i+6] * tempS;
		(*desVec) = *(desVec++) * tempS;//desVec[8*i+7] =  desVec[8*i+7] * tempS;
	}
			/*In order to reduce the influence of non-linear illumination,
			 *a threshold is used to limit the value of element in the unit feature
			 *vector no larger than this threshold. In Z.Wang's work, a value of 0.4 is found
			 *empirically to be a proper threshold.*/
	desVec = desc.data();
	for(short i=0; i<descriptorSize; i++ ){
		if(desVec[i]>0.4){
			desVec[i]=0.4;
		}
	}
	//re-normalize desVec;
	temp = 0;
	for(short i=0; i<descriptorSize; i++){
		temp += desVec[i] * desVec[i];
	}
	temp = 1/sqrt(temp);
	for(short i=0; i<descriptorSize; i++){
		desVec[i] =  desVec[i] * temp;
	}
	return desc;
}    


void BuildMat(vector<segDesc> linesInLeft, vector<segDesc> linesInRight)
{
	double TwoPI = 2 * M_PI;
	const unsigned int numLineLeft = linesInLeft.size();
	const unsigned int numLineRight = linesInRight.size();
	
	nodesList_.clear();
	double angleDif;
	double lengthDif;

	unsigned int dimOfDes = linesInLeft[0].desVec.size();
	double desDisMat[numLineLeft][numLineRight]; //store the descriptor distance of lines in left and right images.

	vector<float> desLeft;
	vector<float> desRight;

	//first compute descriptor distances

	float *desL, *desR, *desMax;

	float minDis, dis, temp;
	for (int idL = 0; idL < numLineLeft; idL++)
	{
		for (int idR = 0; idR < numLineRight; idR++)
		{
			minDis = 100;
			desL = linesInLeft[idL].desVec.data();
			desR = linesInRight[idR].desVec.data();
			desMax = desR + dimOfDes;
			dis = 0;
			while (desR < desMax)
			{
                temp = *(desL++) - *(desR++); //discriptor minus save to temp
				dis += temp * temp;
			}
			dis = sqrt(dis);
			if (dis < minDis)
			{
				minDis = dis;
			}
			desDisMat[idL][idR] = minDis;
		} //end for(int idR=0; idR<rightSize; idR++)
	}	 // end for(int idL=0; idL<leftSize; idL++)


	for (unsigned int i = 0; i < numLineLeft; i++)
	{
		for (unsigned int j = 0; j < numLineRight; j++)
		{
			if (desDisMat[i][j] > DescriptorDifThreshold)
			{
				continue; //the descriptor difference is too large;
			}

			//there doesn't exist a global rotation angle between two image, so the angle difference test is canceled.
			lengthDif = fabs(linesInLeft[i].lineLength - linesInRight[j].lineLength) / MIN(linesInLeft[i].lineLength, linesInRight[j].lineLength);
			if (lengthDif > LengthDifThreshold)
			{
				continue; //the length difference is too large;
			}
			matchNode node; //line i in left image and line j in right image pass the test, (i,j) is a possible matched line pair.
			node.leftLineID = i;
			node.rightLineID = j;
			nodesList_.push_back(node);
		} //end inner loop
	}
	cout << "the number of possible matched line pair = " << nodesList_.size() << endl;
	//	desDisMat.Save("DescriptorDis.txt");

	
	unsigned int dim = nodesList_.size(); // Dimension of the problem.

	//std::array<double, dim_temp> adjacenceVec;
	std::vector<double> adjacenceVec(dim * (dim + 1) / 2, 0);

	int nnz = 0; // Number of nonzero elements in adjacenceMat.
	
	//	Matrix<double> testMat(dim,dim);
	//	testMat.SetZero();

	
    unsigned int bComputedLeft[numLineLeft][numLineLeft]; //flag to show whether the ith pair of left image has already been computed.
    memset(bComputedLeft, 0, numLineLeft * numLineLeft * sizeof(unsigned int));
    double intersecRatioLeft[numLineLeft][numLineLeft]; //the ratio of intersection point and the line in the left pair
    double projRatioLeft[numLineLeft][numLineLeft];		//the point to line distance divided by the projected length of line in the left pair.

    unsigned int bComputedRight[numLineRight][numLineRight]; //flag to show whether the ith pair of right image has already been computed.
    memset(bComputedRight, 0, numLineRight * numLineRight * sizeof(unsigned int));
    double intersecRatioRight[numLineRight][numLineRight]; //the ratio of intersection point and the line in the right pair
    double projRatioRight[numLineRight][numLineRight];	 //the point to line distance divided by the projected length of line in the right pair.

    unsigned int idLeft1, idLeft2;						//the id of lines in the left pair
	unsigned int idRight1, idRight2;					//the id of lines in the right pair
	double relativeAngleLeft, relativeAngleRight;		//the relative angle of each line pair
	double gradientMagRatioLeft, gradientMagRatioRight; //the ratio of gradient magnitude of lines in each pair

	double iRatio1L, iRatio1R, iRatio2L, iRatio2R;
	double pRatio1L, pRatio1R, pRatio2L, pRatio2R;

	double relativeAngleDif, gradientMagRatioDif, iRatioDif, pRatioDif;

	double interSectionPointX, interSectionPointY;
	double a1, a2, b1, b2, c1, c2; //line1: a1 x + b1 y + c1 =0; line2: a2 x + b2 y + c2=0
	double a1b2_a2b1;			   //a1b2-a2b1
	double length1, length2, len;
	double disX, disY;
	double disS, disE;
	double similarity;


	for (unsigned int j = 0; j < dim; j++)
	{ //column
		idLeft1 = nodesList_[j].leftLineID;
		idRight1 = nodesList_[j].rightLineID;
		for (unsigned int i = j + 1; i < dim; i++)
        { //row
			idLeft2 = nodesList_[i].leftLineID;
			idRight2 = nodesList_[i].rightLineID;
			if ((idLeft1 == idLeft2) || (idRight1 == idRight2))
			{
				continue; //not satisfy the one to one match condition
			}
			//first compute the relative angle between left pair and right pair.
			relativeAngleLeft = linesInLeft[idLeft1].segment.angle - linesInLeft[idLeft2].segment.angle;
			relativeAngleLeft = (relativeAngleLeft < M_PI) ? relativeAngleLeft : (relativeAngleLeft - TwoPI);
			relativeAngleLeft = (relativeAngleLeft > (-M_PI)) ? relativeAngleLeft : (relativeAngleLeft + TwoPI);
			relativeAngleRight = linesInRight[idRight1].segment.angle - linesInRight[idRight2].segment.angle;
			relativeAngleRight = (relativeAngleRight < M_PI) ? relativeAngleRight : (relativeAngleRight - TwoPI);
			relativeAngleRight = (relativeAngleRight > (-M_PI)) ? relativeAngleRight : (relativeAngleRight + TwoPI);
			relativeAngleDif = fabs(relativeAngleLeft - relativeAngleRight);
			if ((TwoPI - relativeAngleDif) > RelativeAngleDifferenceThreshold && relativeAngleDif > RelativeAngleDifferenceThreshold)
			{
				continue; //the relative angle difference is too large;
			}
			else if ((TwoPI - relativeAngleDif) < RelativeAngleDifferenceThreshold)
			{
				relativeAngleDif = TwoPI - relativeAngleDif;
			}

			//at last, check the intersect point ratio and point to line distance ratio
			//check whether the geometric information of pairs (idLeft1,idLeft2) and (idRight1,idRight2) have already been computed.
			if (!bComputedLeft[idLeft1][idLeft2])
			{ //have not been computed yet

				a1 = linesInLeft[idLeft1].segment.y2 - linesInLeft[idLeft1].segment.y1;					//disY
				b1 = linesInLeft[idLeft1].segment.x1 - linesInLeft[idLeft1].segment.x2;					//-disX
				c1 = (0 - b1 * linesInLeft[idLeft1].segment.y1) - a1 * linesInLeft[idLeft1].segment.x1; //disX*sy - disY*sx
				length1 = linesInLeft[idLeft1].lineLength;

				a2 = linesInLeft[idLeft2].segment.y2 - linesInLeft[idLeft2].segment.y1;					//disY
				b2 = linesInLeft[idLeft2].segment.x1 - linesInLeft[idLeft2].segment.x2;					//-disX
				c2 = (0 - b2 * linesInLeft[idLeft2].segment.y1) - a2 * linesInLeft[idLeft2].segment.x1; //disX*sy - disY*sx
				length2 = linesInLeft[idLeft2].lineLength;

				a1b2_a2b1 = a1 * b2 - a2 * b1;
				if (fabs(a1b2_a2b1) < 0.001)
				{ //two lines are almost parallel
					iRatio1L = Inf;
					iRatio2L = Inf;
				}
				else
				{
					interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
					interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
					//r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
					disX = interSectionPointX - linesInLeft[idLeft1].segment.x1;
					disY = interSectionPointY - linesInLeft[idLeft1].segment.y1;
					len = disY * a1 - disX * b1;
					iRatio1L = len / (length1 * length1);
					//r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
					disX = interSectionPointX - linesInLeft[idLeft2].segment.x1;
					disY = interSectionPointY - linesInLeft[idLeft2].segment.y1;
					len = disY * a2 - disX * b2;
					iRatio2L = len / (length2 * length2);
				}
				intersecRatioLeft[idLeft1][idLeft2] = iRatio1L;
				intersecRatioLeft[idLeft2][idLeft1] = iRatio2L; //line order changed

				
				disS = fabs(a2 * linesInLeft[idLeft1].segment.x1 + b2 * linesInLeft[idLeft1].segment.y1 + c2) / length2;
				disE = fabs(a2 * linesInLeft[idLeft1].segment.x2 + b2 * linesInLeft[idLeft1].segment.y2 + c2) / length2;
				pRatio1L = (disS + disE) / length1;
				projRatioLeft[idLeft1][idLeft2] = pRatio1L;

				
				disS = fabs(a1 * linesInLeft[idLeft2].segment.x1 + b1 * linesInLeft[idLeft2].segment.y1 + c1) / length1;
				disE = fabs(a1 * linesInLeft[idLeft2].segment.x2 + b1 * linesInLeft[idLeft2].segment.y2 + c1) / length1;
				pRatio2L = (disS + disE) / length2;
				projRatioLeft[idLeft2][idLeft1] = pRatio2L;

				//mark them as computed
				bComputedLeft[idLeft1][idLeft2] = true;
				bComputedLeft[idLeft2][idLeft1] = true;
			}
			else
			{ //read these information from matrix;
				iRatio1L = intersecRatioLeft[idLeft1][idLeft2];
				iRatio2L = intersecRatioLeft[idLeft2][idLeft1];
				pRatio1L = projRatioLeft[idLeft1][idLeft2];
				pRatio2L = projRatioLeft[idLeft2][idLeft1];
			}
            if (!bComputedRight[idRight1][idRight2])
            {
                //have not been computed yet
                a1 = linesInRight[idRight1].segment.y2 - linesInRight[idRight1].segment.y1;					//disY
                b1 = linesInRight[idRight1].segment.x1 - linesInRight[idRight1].segment.x2;					//-disX
                c1 = (0 - b1 * linesInRight[idRight1].segment.y1) - a1 * linesInRight[idRight1].segment.x1; //disX*sy - disY*sx
                length1 = linesInRight[idRight1].lineLength;

                a2 = linesInRight[idRight2].segment.y2 - linesInRight[idRight2].segment.y1;					//disY
                b2 = linesInRight[idRight2].segment.x1 - linesInRight[idRight2].segment.x2;					//-disX
                c2 = (0 - b2 * linesInRight[idRight2].segment.y1) - a2 * linesInRight[idRight2].segment.x1; //disX*sy - disY*sx
                length2 = linesInRight[idRight2].lineLength;

                a1b2_a2b1 = a1 * b2 - a2 * b1;
                if (fabs(a1b2_a2b1) < 0.001)
                { //two lines are almost parallel
                    iRatio1R = Inf;
                    iRatio2R = Inf;
                }
                else
                {
                    interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
                    interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
                    //r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
                    disX = interSectionPointX - linesInRight[idRight1].segment.x1;
                    disY = interSectionPointY - linesInRight[idRight1].segment.y1;
                    len = disY * a1 - disX * b1; //because b1=-disX
                    iRatio1R = len / (length1 * length1);
                    //r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
                    disX = interSectionPointX - linesInRight[idRight2].segment.x1;
                    disY = interSectionPointY - linesInRight[idRight2].segment.y1;
                    len = disY * a2 - disX * b2; //because b2=-disX
                    iRatio2R = len / (length2 * length2);
                }
                intersecRatioRight[idRight1][idRight2] = iRatio1R;
                intersecRatioRight[idRight2][idRight1] = iRatio2R; //line order changed
                
				
                disS = fabs(a2 * linesInRight[idRight1].segment.x1 + b2 * linesInRight[idRight1].segment.y1 + c2) / length2;
                disE = fabs(a2 * linesInRight[idRight1].segment.x2 + b2 * linesInRight[idRight1].segment.y2 + c2) / length2;
                pRatio1R = (disS + disE) / length1;
                projRatioRight[idRight1][idRight2] = pRatio1R;

                
                disS = fabs(a1 * linesInRight[idRight2].segment.x1 + b1 * linesInRight[idRight2].segment.y1 + c1) / length1;
                disE = fabs(a1 * linesInRight[idRight2].segment.x2 + b1 * linesInRight[idRight2].segment.y2 + c1) / length1;
                pRatio2R = (disS + disE) / length2;
                projRatioRight[idRight2][idRight1] = pRatio2R;

                //mark them as computed
                bComputedRight[idRight1][idRight2] = true;
                bComputedRight[idRight2][idRight1] = true;
            }
            else
            { //read these information from matrix;
                iRatio1R = intersecRatioRight[idRight1][idRight2];
                iRatio2R = intersecRatioRight[idRight2][idRight1];
                pRatio1R = projRatioRight[idRight1][idRight2];
                pRatio2R = projRatioRight[idRight2][idRight1];
            }
            pRatioDif = MIN(fabs(pRatio1L - pRatio1R), fabs(pRatio2L - pRatio2R));

			if (pRatioDif > ProjectionRationDifThreshold)
			{
				continue; //the projection length ratio difference is too large;
			}
			if ((iRatio1L == Inf) || (iRatio2L == Inf) || (iRatio1R == Inf) || (iRatio2R == Inf))
            {
				//don't consider the intersection length ratio
				similarity = 4 - desDisMat[idLeft1][idRight1] / DescriptorDifThreshold - desDisMat[idLeft2][idRight2] / DescriptorDifThreshold - pRatioDif / ProjectionRationDifThreshold - relativeAngleDif / RelativeAngleDifferenceThreshold;
                adjacenceVec[(2 * dim - j - 1) * j / 2 + i] = similarity;
				nnz++;
				//				testMat[i][j] = similarity;
				//				testMat[j][i] = similarity;
			}
			else
            {
				iRatioDif = MIN(fabs(iRatio1L - iRatio1R), fabs(iRatio2L - iRatio2R));
				if (iRatioDif > IntersectionRationDifThreshold)
				{
					continue; //the intersection length ratio difference is too large;
				}
				//now compute the similarity score between two line pairs.
				similarity = 5 - desDisMat[idLeft1][idRight1] / DescriptorDifThreshold - desDisMat[idLeft2][idRight2] / DescriptorDifThreshold - iRatioDif / IntersectionRationDifThreshold - pRatioDif / ProjectionRationDifThreshold - relativeAngleDif / RelativeAngleDifferenceThreshold;
				adjacenceVec[(2 * dim - j - 1) * j / 2 + i] = similarity;
				nnz++;
				//				testMat[i][j] = similarity;
				//				testMat[j][i] = similarity;
			}
		}
    }
	// pointer to an array that stores the nonzero elements of Adjacency matrix.
	double *adjacenceMat = new double[nnz];
	// pointer to an array that stores the row indices of the non-zeros in adjacenceMat.
	int *irow = new int[nnz];
	// pointer to an array of pointers to the beginning of each column of adjacenceMat.
	int *pcol = new int[dim + 1];
	int idOfNNZ = 0; //the order of none zero element
	pcol[0] = 0;
	unsigned int tempValue;
	for (unsigned int j = 0; j < dim; j++)
	{ //column
		for (unsigned int i = j; i < dim; i++)
		{ //row
			tempValue = (2 * dim - j - 1) * j / 2 + i;
			if (adjacenceVec[tempValue] != 0)
			{
				adjacenceMat[idOfNNZ] = adjacenceVec[tempValue];
				irow[idOfNNZ] = i;
				idOfNNZ++;
			}
		}
		pcol[j + 1] = idOfNNZ;
    }	
	ARluSymMatrix<double> arMatrix(dim, nnz, adjacenceMat, irow, pcol);
	ARluSymStdEig<double> dprob(2, arMatrix, "LM"); // Defining what we need: the first eigenvector of arMatrix with largest magnitude.
	// Finding eigenvalues and eigenvectors.
	dprob.FindEigenvectors();
    cout << "Number of 'converged' eigenvalues  : " << dprob.ConvergedEigenvalues() << endl;

	eigenMap_.clear();

	double meanEigenVec = 0;
	if (dprob.EigenvectorsFound())
	{
		double value;
		for (unsigned int j = 0; j < dim; j++)
		{
			value = fabs(dprob.Eigenvector(1, j));
			meanEigenVec += value;
			eigenMap_.insert(std::make_pair(value, j));
		}
	}
	minOfEigenVec_ = WeightOfMeanEigenVec * meanEigenVec / dim;
	delete[] adjacenceMat;
	delete[] irow;
	delete[] pcol;
}

void MatchingResult(vector<segDesc> linesInLeft, vector<segDesc> linesInRight, vector<unsigned int> &matchResult)
{
	double TwoPI = 2 * M_PI;
	vector<unsigned int> matchRet1;
	vector<unsigned int> matchRet2;
	double matchScore1 = 0;
	double matchScore2 = 0;
	EigenMAP mapCopy = eigenMap_;
	unsigned int dim = nodesList_.size();
	EigenMAP::iterator iter;
	unsigned int id, idLeft2, idRight2;
	double sideValueL, sideValueR;
	double pointX, pointY;
	double relativeAngleLeft, relativeAngleRight; //the relative angle of each line pair
	double relativeAngleDif;

	//store eigenMap for debug
	fstream resMap;
	ostringstream fileNameMap;
	fileNameMap << "eigenVec.txt";
	resMap.open(fileNameMap.str().c_str(), std::ios::out);

	double mat[linesInLeft.size()][linesInRight.size()];
	memset(mat, 0, linesInLeft.size() * linesInRight.size());
	for (iter = eigenMap_.begin(); iter != eigenMap_.end(); iter++)
	{
		id = iter->second;
		resMap << nodesList_[id].leftLineID << "    " << nodesList_[id].rightLineID << "   " << iter->first << endl;
		mat[nodesList_[id].leftLineID][nodesList_[id].rightLineID] = iter->first;
	}
	//mat.Save("eigenMap.txt");
	// matSave("eigenMap.txt");
	resMap.flush();
	resMap.close();

	/*first try, start from the top element in eigenmap */
	while (1)
	{
		iter = eigenMap_.begin();
		//if the top element in the map has small value, then there is no need to continue find more matching line pairs;
		if (iter->first < minOfEigenVec_)
		{
			break;
		}
		id = iter->second;
		unsigned int idLeft1 = nodesList_[id].leftLineID;
		unsigned int idRight1 = nodesList_[id].rightLineID;
		matchRet1.push_back(idLeft1);
		matchRet1.push_back(idRight1);
		matchScore1 += iter->first;
		eigenMap_.erase(iter++);
		//remove all potential assignments in conflict with top matched line pair
		double xe_xsLeft = linesInLeft[idLeft1].segment.x2 - linesInLeft[idLeft1].segment.x1;
		double ye_ysLeft = linesInLeft[idLeft1].segment.y2 - linesInLeft[idLeft1].segment.y1;
		double xe_xsRight = linesInRight[idRight1].segment.x2 - linesInRight[idRight1].segment.x1;
		double ye_ysRight = linesInRight[idRight1].segment.y2 - linesInRight[idRight1].segment.y1;
		double coefLeft = sqrt(xe_xsLeft * xe_xsLeft + ye_ysLeft * ye_ysLeft);
		double coefRight = sqrt(xe_xsRight * xe_xsRight + ye_ysRight * ye_ysRight);
		for (; iter->first >= minOfEigenVec_;)
		{
			id = iter->second;
			idLeft2 = nodesList_[id].leftLineID;
			idRight2 = nodesList_[id].rightLineID;
			//check one to one match condition
			if ((idLeft1 == idLeft2) || (idRight1 == idRight2))
			{
				eigenMap_.erase(iter++);
				continue; //not satisfy the one to one match condition
			}
			//check sidedness constraint, the middle point of line2 should lie on the same side of line1.
			//sideValue = (y-ys)*(xe-xs)-(x-xs)*(ye-ys);
			pointX = 0.5 * (linesInLeft[idLeft2].segment.x1 + linesInLeft[idLeft2].segment.x2);
			pointY = 0.5 * (linesInLeft[idLeft2].segment.y1 + linesInLeft[idLeft2].segment.y2);
			sideValueL = (pointY - linesInLeft[idLeft1].segment.y1) * xe_xsLeft - (pointX - linesInLeft[idLeft1].segment.x1) * ye_ysLeft;
			sideValueL = sideValueL / coefLeft;
			pointX = 0.5 * (linesInRight[idRight2].segment.x1 + linesInRight[idRight2].segment.x2);
			pointY = 0.5 * (linesInRight[idRight2].segment.y1 + linesInRight[idRight2].segment.y2);
			sideValueR = (pointY - linesInRight[idRight1].segment.y1) * xe_xsRight - (pointX - linesInRight[idRight1].segment.x1) * ye_ysRight;
			sideValueR = sideValueR / coefRight;
			if (sideValueL * sideValueR < 0 && fabs(sideValueL) > 5 && fabs(sideValueR) > 5)
			{ //have the different sign, conflict happens.
				eigenMap_.erase(iter++);
				continue;
			}
			//check relative angle difference
			relativeAngleLeft = linesInLeft[idLeft1].segment.angle - linesInLeft[idLeft2].segment.angle;
			relativeAngleLeft = (relativeAngleLeft < M_PI) ? relativeAngleLeft : (relativeAngleLeft - TwoPI);
			relativeAngleLeft = (relativeAngleLeft > (-M_PI)) ? relativeAngleLeft : (relativeAngleLeft + TwoPI);
			relativeAngleRight = linesInRight[idRight1].segment.angle - linesInRight[idRight2].segment.angle;
			relativeAngleRight = (relativeAngleRight < M_PI) ? relativeAngleRight : (relativeAngleRight - TwoPI);
			relativeAngleRight = (relativeAngleRight > (-M_PI)) ? relativeAngleRight : (relativeAngleRight + TwoPI);
			relativeAngleDif = fabs(relativeAngleLeft - relativeAngleRight);
			if ((TwoPI - relativeAngleDif) > RelativeAngleDifferenceThreshold && relativeAngleDif > RelativeAngleDifferenceThreshold)
			{
				eigenMap_.erase(iter++);
				continue; //the relative angle difference is too large;
			}
			iter++;
		}
	} //end while(stillLoop)
	matchResult = matchRet1;
	cout << "matchRet1.size=" << matchRet1.size() << ", minOfEigenVec_= " << minOfEigenVec_ << endl;
}    