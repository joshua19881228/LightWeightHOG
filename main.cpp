#include "CTables.h"
#include "CHog.h"
#include "time.h"
#include "stdio.h"
#include <sstream>

int NUM_ORI_HALF = 9;  //方向量化bin数//
int NUM_ORI_FULL = 18;
int CELL_SIZE = 8;  //HOG中cell尺寸，只实现了正方形cell//
double SUB_RATIO = 2;
int WINDOW_WIDTH = 6;
int WINDOW_HEIGHT =14;

#define IMAGE_SRC
#ifdef IMAGE_SRC
int main(int argc, char** argv)
{
    string imagePath;
    string exportPath;
    if(argc==2)
    {
        imagePath = argv[1];
    }
    else
    {
        cout<<"wrong input"<<endl;
        exit(0);
    }
    CTables tables(NUM_ORI_HALF, NUM_ORI_FULL, CELL_SIZE, true);  //建立Look Up Table
    CHog hog;

    cv::Mat cvImage = cv::imread(imagePath);
	if (cvImage.data == NULL)
	{
		cout << "no such image: " << "imagePath" << endl;
		exit(0);
	}
    int width = cvImage.cols/CELL_SIZE*CELL_SIZE;
    int height = cvImage.rows/CELL_SIZE*CELL_SIZE;
    hog.initialHist(width,
                height,
                cvImage.channels(),
                NUM_ORI_HALF,
                NUM_ORI_FULL,
                CELL_SIZE);  //初始化一幅图像的HOG直方图
    cv::resize(cvImage, cvImage, cv::Size(width, height));  //将图像尺寸调整到cell尺寸的整倍数
    printf("width:%d, height:%d\n", cvImage.cols, cvImage.rows);
    unsigned char* image = cvImage.ptr();  //将图像数据传给指针，不知道去除opencv的数据类型是否有助于提升速度

    hog.calcDiff(image, tables.magnitudeTable);
    hog.calcHist(tables.orientationFullTable,
                 tables.orientationHalfTable,
                 tables.interpolationTableInt,
                 tables.weightedBinTable);  //计算HOG直方图

	cv::Mat visualMat = hog.visualizationFull();
	//cv::imshow("hog mat", visualMat);
	//cv::waitKey(0);
    exportPath = "./" + imagePath.substr(imagePath.rfind("/")+1, imagePath.rfind(".") - imagePath.rfind("/") - 1) + "_hog.png";
    cv::imwrite(exportPath, visualMat);
	return 0;
}
#else
int main(int argc, char** argv)
{
    string videoPath = "test.avi";  //视频路径
    //if(argc==2)
    //{
    //    videoPath = argv[1];
    //}

    CTables tables(NUM_ORI_HALF, NUM_ORI_FULL, CELL_SIZE, true);  //建立Look Up Table

    clock_t begin, finish;  //用于粗精度计时
    cv::Mat cvImage;
    unsigned char* image;
    int nframe = 0, savedNum = 0;
    int width, height;
	int p;
    CHog hog;

	cv::VideoCapture cap(videoPath);
    if(!cap.isOpened())
    {
        cout<<"video is not opened"<<endl;
        exit(0);
    }
    width = cap.get(CV_CAP_PROP_FRAME_WIDTH)/SUB_RATIO;
    width = width/CELL_SIZE*CELL_SIZE;
    height = cap.get(CV_CAP_PROP_FRAME_HEIGHT)/SUB_RATIO;
    height = height/CELL_SIZE*CELL_SIZE;
    printf("width:%d, height:%d\n", width, height);

    hog.initialHist(width,
                height,
                3,
                NUM_ORI_HALF,
                NUM_ORI_FULL,
                CELL_SIZE);  //初始化一幅图像的HOG直方图

    while(cap.read(cvImage))
    {
        nframe++;
        cv::resize(cvImage, cvImage, cv::Size(width, height));  //将图像尺寸调整到cell尺寸的整倍数
        image = cvImage.ptr();  //将图像数据传给指针，不知道去除opencv的数据类型是否有助于提升速度

        begin = clock();
        hog.clearHist();

        hog.calcDiff(image, tables.magnitudeTable);
        hog.calcHist(tables.orientationFullTable,
                     tables.orientationHalfTable,
                     tables.interpolationTableInt,
                     tables.weightedBinTable);  //计算HOG直方图

        finish = clock();
        cout<<"time cost: "<<(double)(finish - begin)/CLOCKS_PER_SEC<<" sec"<<endl;
        cv::Mat cvVisualizationHalf = hog.visualizationHalf();  //将直方图用图像进行显示
		cv::imshow("hog", cvVisualizationHalf);
		cv::imshow("src", cvImage);
		p = cv::waitKey(1);
		if (p == 27)
			break;
    }
    return 0;
}

#endif
