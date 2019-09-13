# Multiple Homography Estimation via Stereo Line Matching for Textureless Indoor Scenes

I have done the oral presentation on this topic on the International Conference on Control, Automation and Robotics(ICCAR2019). This paper can be found online at: https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=iccar&tdsourcetag=s_pctim_aiomsg.

The major contributions of this paper are:

1. Developed a unified approach for line detection and stereo matching that can produce appealing results in indoor textureless scenes.

2. Proposed a co-planar line classification algorithm.

3. Pointed out that the compatibility of multiple homographies can be improved by enforcing the epipolar line constraints.

The code here is our hybrid algorithm for line detection and stereo matching.

In https://github.com/slslam/slslam, J. Lee, S. Lee, G. Zhang, J. Lim and I. Suh proposed a Canny-based line extractor and integrated it with the MSLD descriptor for line detection and matching. It is shown that this line detection algorithm is capable of detecting weak textures as well as generating non-fragmented line segments.

In https://github.com/mtamburrano/LBD_Descriptor, L. Zhang and R. Koch proposed the LBD descriptor for line matching. According to them, LBD descriptor is more efficient to compute and it is faster to generate the matching results than the state-of-the-art methods.

However, there are a couple of drawbacks of the originally proposed LBD-based approach. First, in the originally proposed LBD-based approach, LBD descriptor is incorporated with EDLine detector and the latter is proved to be inefficient to detect low-contrast edges in our experiments. Second, in the originally proposed LBD-based approach, efforts are made to overcome the scale changes and the global rotation, which won't produce any improvement in our cases given that we apply the algorithm to calibrated, stereo cameras.

Our code has the advantages of both the Canny+MSLD approach and the EDLine+LBD approach and it can produce superior results in the indoor textureless scenes.
