#include "CTables.h"
#include "CHog.h"
//#include "CSVM.h"
//#include "postProcess.h"
#include "time.h"
#include "stdio.h"
#include <sstream>

#if _DEBUG
	#pragma comment(lib,"opencv_core2410d.lib")
	#pragma comment(lib,"opencv_highgui2410d.lib")
	#pragma comment(lib,"opencv_imgproc2410d.lib")
#else
	#pragma comment(lib,"opencv_core2410.lib")
	#pragma comment(lib,"opencv_highgui2410.lib")
	#pragma comment(lib,"opencv_imgproc2410.lib")
#endif


int NUM_ORI_HALF = 9;  //方向量化bin数//
int NUM_ORI_FULL = 18;
int CELL_SIZE = 8;  //HOG中cell尺寸，只实现了正方形cell//
double SUB_RATIO = 2;
int WINDOW_WIDTH = 6;
int WINDOW_HEIGHT =14;

//#define IMAGE_SRC
#ifdef IMAGE_SRC
int main(int argc, char** argv)
{
    string imagePath;
    string exportPath;
    int label;
    if(argc==4)
    {
        imagePath = argv[1];
        exportPath = argv[2];
        label = atoi(argv[3]);
    }
    else if(argc==1)
    {
        imagePath = "test.jpg";  //图像路径
        exportPath = ".";
        label = -2;
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
	cv::imshow("hog mat", visualMat);
	cv::waitKey(0);
	
    //CSVM svm(31*WINDOW_HEIGHT*WINDOW_WIDTH);
    //svm.loadModel("hog_ped_weight.txt");
    //double* feat = new double[31*WINDOW_HEIGHT*WINDOW_WIDTH];
    //hog.slideWindowPredict(feat, svm, cvImage);
    //delete[] feat;

    //string exportPathNeg = exportPath+"/negFeat.txt";
    //string exportPathPos = exportPath+"/posFeat.txt";
    //if(label==-1)
    //    hog.cropNegSample(cvImage, -1, exportPathNeg);
    //else if(label==-2)
    //{
    //    hog.cropNegSample(cvImage, exportPathNeg);
    //}
    //else if(label == 1)
    //{
    //    hog.cropPosSample(cvImage, +1, exportPathPos);
    //    hog.cropNegSample(cvImage, -1, exportPathNeg);
    //}
    //else
    //{
    //    cout<<"usage: run imagePath exportPath label"<<endl
    //    <<"extract negative sample from cropped image: -1"<<endl
    //    <<"extract postive and negtive sample from cropped image: 1"<<endl
    //    <<"extract negative sample from whole image: -2"<<endl;
    //}

    return 0;
}
#else
int main(int argc, char** argv)
{
    string videoPath = "test.avi";  //视频路径
    //string modelPath = "hog_ped_weight.txt";
    //string saveRoot = "./crop";
    int label = 1;
    //if(argc==3)
    //{
    //    videoPath = argv[1];
    //    modelPath = argv[2];
    //}
    //else if(argc==5)
    //{
    //    videoPath = argv[1];
    //    modelPath = argv[2];
    //    saveRoot = argv[3];
    //    label = atoi(argv[4]);
    //}
    //else
    //{
    //    cout<<"usage: run videoPath modelPath [saveRoot label]"<<endl;
        //exit(0);
    //}

    CTables tables(NUM_ORI_HALF, NUM_ORI_FULL, CELL_SIZE, true);  //建立Look Up Table
    //CSVM svm(31*WINDOW_HEIGHT*WINDOW_WIDTH);

    clock_t begin, finish;  //用于粗精度计时
    cv::Mat cvImage;
    unsigned char* image;
    int nframe = 0, savedNum = 0;
    int width, height;
	int p;
    CHog hog;
    //double* feat = new double[31*WINDOW_HEIGHT*WINDOW_WIDTH];
    //double score;
    //vector<boundingBox> boundingBoxes(0);
    //boundingBox aBoundingBox;

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

    //svm.loadModel(modelPath);
    hog.initialHist(width,
                height,
                3,
                NUM_ORI_HALF,
                NUM_ORI_FULL,
                CELL_SIZE);  //初始化一幅图像的HOG直方图
    //cv::Mat scoreMap(height, width, CV_8SC3);

    while(cap.read(cvImage))
    {
        nframe++;
        cv::resize(cvImage, cvImage, cv::Size(width, height));  //将图像尺寸调整到cell尺寸的整倍数
        image = cvImage.ptr();  //将图像数据传给指针，不知道去除opencv的数据类型是否有助于提升速度

        begin = clock();
        hog.clearHist();
        //boundingBoxes.clear();

        hog.calcDiff(image, tables.magnitudeTable);
        hog.calcHist(tables.orientationFullTable,
                     tables.orientationHalfTable,
                     tables.interpolationTableInt,
                     tables.weightedBinTable);  //计算HOG直方图

		/*
        for(int y = 0; y < height/CELL_SIZE; y++)
        {
            for(int x = 0; x < width/CELL_SIZE; x++)
            {
                //cout<<"y:"<<y<<" x:"<<x<<endl;
                //hog.exportFeatVect(exportPath, x, y, 6, 14, label);
                hog.getFeatVect(feat, x, y, WINDOW_WIDTH, WINDOW_HEIGHT);
                score = svm.predict(feat);
                cv::Point uls(x*CELL_SIZE, y*CELL_SIZE);
                cv::Point rbs((x+1)*CELL_SIZE, (y+1)*CELL_SIZE);
                //cout<<"score: "<<score<<endl;
                if(score>2)
                {
                    cv::rectangle(scoreMap, uls, rbs, cv::Scalar(0, 0, min(int(score/2.5*255), 255)), -1);
                    aBoundingBox.subRatio = SUB_RATIO;
                    aBoundingBox.x = uls.x;
                    aBoundingBox.y = uls.y;
                    aBoundingBox.width = WINDOW_WIDTH*CELL_SIZE;
                    aBoundingBox.height = WINDOW_HEIGHT*CELL_SIZE;
                    aBoundingBox.score = score;
                    boundingBoxes.push_back(aBoundingBox);
                }
                else
                    cv::rectangle(scoreMap, uls, rbs, cv::Scalar(min(int(-score/2.5*255), 255), 0, 0), -1);
            }
        }
        NMS(boundingBoxes);
        saveVisuaBB(cvImage, boundingBoxes, saveRoot, savedNum);
        visualizeBB(cvImage, boundingBoxes);
		*/
        finish = clock();
        cout<<"time cost: "<<(double)(finish - begin)/CLOCKS_PER_SEC<<" sec"<<endl;
        cv::Mat cvVisualizationHalf = hog.visualizationHalf();  //将直方图用图像进行显示
		cv::imshow("hog", cvVisualizationHalf);
		cv::imshow("src", cvImage);
        //cv::imshow("image", cvImage);
        //cv::imshow("visualScore", scoreMap);
		p = cv::waitKey(1);
		if (p == 27)
			break;
    }
    //delete[] feat;
    return 0;
}

#endif
