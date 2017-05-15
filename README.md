# LightWeightHOG #

A fast HOG feature extractor is implemented in this [repo](https://github.com/joshua19881228/LightWeightHOG). This code for extracting HOG features in images is a naive implementation based on the work of [The Fastest Deformable Part Model for Object Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Yan_The_Fastest_Deformable_2014_CVPR_paper.pdf), which is published in CVPR2014. Note that only the feature extraction part is implemented.

It was every interesting that I wrote this HOG extractor on a Saturday in 2014. My roomate went to his company and locked me home. I could not open the door, having nothing to do. Then I thought why not implementing this fast HOG extractor. So here it is. 

| Input | Output |
| ----- | ------ |
| ![input](https://raw.githubusercontent.com/joshua19881228/LightWeightHOG/master/test.jpg){width:20;} | ![output](https://raw.githubusercontent.com/joshua19881228/LightWeightHOG/master/test_hog.png) |


## Reference ##

Yan, Junjie, et al. "The Fastest Deformable Part Model for Object Detection." IEEE Conference on Computer Vision and Pattern Recognition IEEE, 2014:2497-2504.
