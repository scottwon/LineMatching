#include "LineMatchingAlgorithm.hpp"

using namespace std;

void usage(int argc, char **argv)
{
	cout << "Usage: " << argv[0] << "  image1.png"
		 << "  image2.png" << " out.png" << endl;
}

int main(int argc, char **argv)
{
	int ret = -1;
	if (argc < 4)
	{
		usage(argc, argv);
		return ret;
	}
    //load first image from file
	double t = (double)cv::getTickCount();
    image_process(argv[1],argv[2],argv[3],argv[4],false);
	double t2 = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
	cout<<t2<<endl;
    return 0;
}
