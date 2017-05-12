#ifndef CHOG_H_INCLUDED
#define CHOG_H_INCLUDED

#include "CTables.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
const int KERNEL[3] = {-1, 0 ,1};  //求解梯度所用核

//HOG中cell结构体//
struct histCell
{
    double* cellBinsHalf;  //cell中的直方图
    double* cellBinsFull;
    double* feat;
    int numOriHalf;  //直方图的bin数，即方向量化个数
    int numOriFull;
    int dim;
    double normFactorHalf;
    //double normFactorFull;
};

//HOG类//
class CHog
{
public:
    CHog();  //构造函数
    ~CHog();  //析构函数
    void initialHist(int imgWidth,
                     int imgHeight,
                     int imgChannel,
                     int windowWidth,
                     int windowHeight,
                     int numOriHalf,
                     int numOriFull,
                     int cellSize);  //初始化直方图
    void initialHist(int imgWidth,
                     int imgHeight,
                     int imgChannel,
                     int numOriHalf,
                     int numOriFull,
                     int cellSize);  //初始化直方图
    void clearHist();
    void clearDiff();
    void calcHist(int windowX,
                  int windowY,
                  int** orientationFullTable, //梯度方向LUT
                  int** orientationHalfTable,
                  interpItemInt** interpolationTableInt, //内差权重LUT
                  double** weightedBinTable);  //计算直方图
    void calcHist(int** orientationFullTable, //梯度方向LUT
                  int** orientationHalfTable,
                  interpItemInt** interpolationTableInt, //内差权重LUT
                  double** weightedBinTable);  //计算直方图
    void calcDiff(unsigned char* image, double** magnitudeTable);
    cv::Mat visualizationHalf(cv::Mat image = cv::Mat());  //用图像显示HOG直方图，基于OPENCV
    cv::Mat visualizationFull(cv::Mat image = cv::Mat());  //用图像显示HOG直方图，基于OPENCV
    cv::Mat visualizeWindow(int x, int y, int width, int height);
    void saveHist(string outPath);  //保存HOG直方图
    void saveHistMax(string outPath);  //保存各个cell中的最大值
    void exportFeatVect(string exportPath, int x, int y, int w, int h, int l);
    void getFeatVect(double* featVect, int x, int y, int w, int h);
private:
    void releaseHist();  //释放直方图空间
    void releaseDiff();
    inline double mmax(double a, double b);
public:
    histCell** hist;  //HOG直方图
    int **dx, **dy;
private:
    int numCellX;  //图像x方向包含的cell个数
    int numCellY;  //图像y方向包含的cell个数
    int cellSize;  //cell的尺寸
    //为了方便计算图像边缘处的直方图，在最外围多申请一圈cell的空间，避免进行边缘检验
    int histSizeX;  //histSizeX = numCellX + 2，左右各多出一个cell用于处理位于图像边缘位置的cell
    int histSizeY;  //histSizeY = numCellY + 2，上下各多出一个cell用于处理位于图像边缘位置的cell
    int imgWidth;
    int imgHeight;
    int windowWidth;
    int windowHeight;
    int imgChannel;
    int widthStep;
};

CHog::CHog()
{
    hist = NULL;  //将直方图至空
    dx = NULL;
    dy = NULL;
}

CHog::~CHog()
{
    releaseHist();  //释放直方图空间
    releaseDiff();
}

inline double CHog::mmax(double a, double b)
{
    double x = a>b ? a:b;
    return x;
}

//初始化直方图//
void CHog::initialHist(int _imgWidth,
                       int _imgHeight,
                       int _imgChannel,
                       int _windowWidth,
                       int _windowHeight,
                       int _numOriHalf, //方向量化个数
                       int _numOriFull,
                       int _cellSize) //cell尺寸
{
    //cout<<"hog hist intializing"<<endl;
    if(_windowWidth%_cellSize!=0 || _windowHeight%_cellSize!=0)  //判断图像尺寸是否是cell尺寸的整倍数
    {
        cout<<"image size is not N*cellSize"<<endl;
        exit(0);
    }
    cellSize = _cellSize;
    windowHeight = _windowHeight;
    windowWidth = _windowWidth;
    numCellX = _windowWidth/cellSize;
    numCellY = _windowHeight/cellSize;
    histSizeX = numCellX + 2;
    histSizeY = numCellY + 2;
    imgWidth = _imgWidth;
    imgHeight = _imgHeight;
    imgChannel = _imgChannel;
    widthStep = imgWidth*imgChannel;
    //为直方图分配空间，并赋初始值0//
    hist = new histCell*[histSizeY];
    for(int y=0; y<histSizeY; y++)
    {
        hist[y] = new histCell[histSizeX];
        for(int x=0; x<histSizeX; x++)
        {
            hist[y][x].cellBinsFull = new double[_numOriFull];  //为一个cell进行空间分配
            hist[y][x].cellBinsHalf = new double[_numOriHalf];
            hist[y][x].dim = 4 + (_numOriFull+_numOriHalf);
            hist[y][x].feat = new double[hist[y][x].dim];
            hist[y][x].numOriFull = _numOriFull;
            hist[y][x].numOriHalf = _numOriHalf;
            //hist[y][x].normFactorFull = 0.;
            hist[y][x].normFactorHalf = 0.;
            memset(hist[y][x].cellBinsFull, 0, hist[y][x].numOriFull*sizeof(double));  //将直方图初始化为全0
            memset(hist[y][x].cellBinsHalf, 0, hist[y][x].numOriHalf*sizeof(double));
            memset(hist[y][x].feat, 0, hist[y][x].dim*sizeof(double));
        }
    }
    //为微分图像分配空间,并赋初始值0//
    dx = new int*[imgHeight];
    dy = new int*[imgHeight];
    for(int y=0; y<imgHeight; y++)
    {
        dx[y] = new int[imgWidth];
        dy[y] = new int[imgWidth];
        memset(dx[y], 0, imgWidth*sizeof(int));
        memset(dy[y], 0, imgWidth*sizeof(int));
    }
    //cout<<"hog hist intialized"<<endl;
}

void CHog::initialHist(int _imgWidth,
                       int _imgHeight,
                       int _imgChannel,
                       int _numOriHalf, //方向量化个数
                       int _numOriFull,
                       int _cellSize) //cell尺寸
{
    //cout<<"hog hist intializing"<<endl;
    if(_imgWidth%_cellSize!=0 || _imgHeight%_cellSize!=0)  //判断图像尺寸是否是cell尺寸的整倍数
    {
        cout<<"image size is not N*cellSize"<<endl;
        exit(0);
    }
    cellSize = _cellSize;
    numCellX = _imgWidth/cellSize;
    numCellY = _imgHeight/cellSize;
    histSizeX = numCellX + 2;
    histSizeY = numCellY + 2;
    imgWidth = _imgWidth;
    imgHeight = _imgHeight;
    imgChannel = _imgChannel;
    widthStep = imgWidth*imgChannel;
    windowWidth = 6;
    windowHeight = 14;
    //为直方图分配空间，并赋初始值0//
    hist = new histCell*[histSizeY];
    for(int y=0; y<histSizeY; y++)
    {
        hist[y] = new histCell[histSizeX];
        for(int x=0; x<histSizeX; x++)
        {
            hist[y][x].cellBinsFull = new double[_numOriFull];  //为一个cell进行空间分配
            hist[y][x].cellBinsHalf = new double[_numOriHalf];
            hist[y][x].dim = 4 + (_numOriFull+_numOriHalf);
            hist[y][x].feat = new double[hist[y][x].dim];
            hist[y][x].numOriFull = _numOriFull;
            hist[y][x].numOriHalf = _numOriHalf;
            //hist[y][x].normFactorFull = 0.;
            hist[y][x].normFactorHalf = 0.;
            memset(hist[y][x].cellBinsFull, 0, hist[y][x].numOriFull*sizeof(double));  //将直方图初始化为全0
            memset(hist[y][x].cellBinsHalf, 0, hist[y][x].numOriHalf*sizeof(double));
            memset(hist[y][x].feat, 0, hist[y][x].dim*sizeof(double));
        }
    }
    //为微分图像分配空间,并赋初始值0//
    dx = new int*[imgHeight];
    dy = new int*[imgHeight];
    for(int y=0; y<imgHeight; y++)
    {
        dx[y] = new int[imgWidth];
        dy[y] = new int[imgWidth];
        memset(dx[y], 0, imgWidth*sizeof(int));
        memset(dy[y], 0, imgWidth*sizeof(int));
    }
    //cout<<"hog hist intialized"<<endl;
}

void CHog::clearHist()
{
    //cout<<"hog hist clearing"<<endl;
    //为直方图赋初始值//
    for(int y=0; y<histSizeY; y++)
    {
        for(int x=0; x<histSizeX; x++)
        {
            memset(hist[y][x].cellBinsFull, 0, hist[y][x].numOriFull*sizeof(double));  //将直方图初始化为全0
            memset(hist[y][x].cellBinsHalf, 0, hist[y][x].numOriHalf*sizeof(double));  //将直方图初始化为全0
            memset(hist[y][x].feat, 0, hist[y][x].dim*sizeof(double));
            //hist[y][x].normFactorFull = 0.;
            hist[y][x].normFactorHalf = 0.;
        }
    }
    //cout<<"hog hist cleared"<<endl;
}

void CHog::clearDiff()
{
    //cout<<"clearing differential"<<endl;
    for(int y=0; y<imgHeight; y++)
    {
        memset(dx[y], 0, imgWidth*sizeof(int));
        memset(dy[y], 0, imgWidth*sizeof(int));
    }
}

//释放直方图空间//
void CHog::releaseHist()
{
    for(int y=0; y<histSizeY; y++)
    {
        for(int x=0; x<histSizeX; x++)
        {
            delete[] hist[y][x].cellBinsFull; //施放一个cell的空间
            delete[] hist[y][x].cellBinsHalf; //施放一个cell的空间
            delete[] hist[y][x].feat;
        }
        delete[] hist[y];
    }
    delete[] hist;
}

void CHog::releaseDiff()
{
    for(int y=0; y<imgHeight; y++)
    {
        delete[] dx[y];
        delete[] dy[y];
    }
    delete[] dx;
    delete[] dy;
}

//计算图像微分//
void CHog::calcDiff(unsigned char* image, double** magnitudeTable) //图像数据
{
    int x=0, y=0, c=0;
    int xDiffs, yDiffs, magni;
    //cout<<"calculating differential"<<endl;
    //遍历图像//
    for(y=0; y<imgHeight; y++)
    {
        for(x=0; x<imgWidth; x++)
        {
            dx[y][x] = (int)image[y*widthStep+((x-1+imgWidth)%imgWidth)*imgChannel+c] - (int)image[y*widthStep+((x+1+imgWidth)%imgWidth)*imgChannel+c]; //x方向梯度
            dy[y][x] = (int)image[(y-1+imgHeight)%imgHeight*widthStep+x*imgChannel+c] - (int)image[(y+1+imgHeight)%imgHeight*widthStep+x*imgChannel+c]; //y方向梯度
            magni = magnitudeTable[dy[y][x]+255][dx[y][x]+255];
            for(c=1; c<imgChannel; c++)
            {
                xDiffs = (int)image[y*widthStep+((x-1+imgWidth)%imgWidth)*imgChannel+c] - (int)image[y*widthStep+((x+1+imgWidth)%imgWidth)*imgChannel+c]; //x方向梯度
                yDiffs = (int)image[(y-1+imgHeight)%imgHeight*widthStep+x*imgChannel+c] - (int)image[(y+1+imgHeight)%imgHeight*widthStep+x*imgChannel+c]; //y方向梯度
                if(magni < magnitudeTable[yDiffs+255][yDiffs+255])
                {
                    dx[y][x] = xDiffs;
                    dy[y][x] = yDiffs;
                }
            }

        }
    }
    //cout<<"differential calculated\t"<<endl;
}

//计算HOG直方图//
void CHog::calcHist(int windowX,
                    int windowY,
                    int** orientationFullTable, //梯度方向LUT
                    int** orientationHalfTable,
                    interpItemInt** interpolationTableInt, //内差权重LUT
                    double** weightedBinTable) //权重×梯度强度LUT
{
    int x=0, y=0, n;
    int xDiff, yDiff, oriIdxFull, oriIdxHalf;
    int cellCoorX, cellCoorY, cellX, cellY, dstCellCoorX, dstCellCoorY, weight;
    int xEnd = windowX + windowWidth;
    int yEnd = windowY + windowHeight;
    int setoff = hist[0][0].numOriHalf+hist[0][0].numOriFull;
    double weightedBinValue;
    double N[4] = {0.}, tmp[4];
    if(xEnd>imgWidth || yEnd>imgHeight)
    {
        cout<<"window out of image"<<endl;
        exit(0);
    }

    //cout<<"calculating hist"<<endl;
    //遍历图像//
    for(y=windowY; y<yEnd; y++)
    {
        for(x=windowX; x<xEnd; x++)
        {
            xDiff = dx[y][x]; //x方向梯度
            yDiff = dy[y][x]; //y方向梯度
            oriIdxFull = orientationFullTable[yDiff+255][xDiff+255]; //量化后方向索引
            oriIdxHalf = orientationHalfTable[yDiff+255][xDiff+255];
            cellCoorY = (y-windowY)/cellSize + 1; //cell在hist中的y坐标
            cellCoorX = (x-windowX)/cellSize + 1; //cell在hist中的x坐标
            cellY = y-windowY - (cellCoorY-1)*cellSize; //图像像素在cell中的y坐标
            cellX = x-windowX - (cellCoorX-1)*cellSize; //图像像素在cell中的x坐标
            //遍历4个需要进行直方图累加的cell
            for(n=0; n<4; n++)
            {
                //计算内差cell在hist中的坐标
                dstCellCoorY = cellCoorY + interpolationTableInt[cellY][cellX].displace[n][1];
                dstCellCoorX = cellCoorX + interpolationTableInt[cellY][cellX].displace[n][0];
                weight = interpolationTableInt[cellY][cellX].weightRatio[n]; //查表获取内差权重
                weightedBinValue = weightedBinTable[weight][(yDiff+255)*NUM_DIFF + (xDiff+255)]; //查表获取内差值
                hist[dstCellCoorY][dstCellCoorX].cellBinsFull[oriIdxFull] += weightedBinValue; //计算直方图
                hist[dstCellCoorY][dstCellCoorX].cellBinsHalf[oriIdxHalf] += weightedBinValue;
            }
        }
    }

    for(cellY=0; cellY<histSizeY; cellY++)
    {
        for(cellX=0; cellX<histSizeX; cellX++)
        {
            for(n=0; n<hist[cellY][cellX].numOriHalf; n++)
                hist[cellY][cellX].normFactorHalf += hist[cellY][cellX].cellBinsHalf[n%hist[cellY][cellX].numOriFull]*hist[cellY][cellX].cellBinsHalf[n%hist[cellY][cellX].numOriFull];
        }
    }

    for(cellY=1; cellY<numCellY+1; cellY++)
    {
        for(cellX=1; cellX<numCellX+1; cellX++)
        {
            N[0] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY-1][cellX].normFactorHalf
                     + hist[cellY][cellX-1].normFactorHalf + hist[cellY-1][cellX-1].normFactorHalf);
            N[1] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY+1][cellX].normFactorHalf
                     + hist[cellY][cellX-1].normFactorHalf + hist[cellY+1][cellX-1].normFactorHalf);
            N[2] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY+1][cellX].normFactorHalf
                     + hist[cellY][cellX+1].normFactorHalf + hist[cellY+1][cellX+1].normFactorHalf);
            N[3] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY-1][cellX].normFactorHalf
                     + hist[cellY][cellX+1].normFactorHalf + hist[cellY-1][cellX+1].normFactorHalf);
            for(n=0; n<hist[cellY][cellX].numOriHalf; n++)
            {
                tmp[0] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[0], 0.2);
                tmp[1] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[1], 0.2);
                tmp[2] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[2], 0.2);
                tmp[3] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[3], 0.2);
                hist[cellY][cellX].feat[n] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                hist[cellY][cellX].feat[setoff + 0] += tmp[0];
                hist[cellY][cellX].feat[setoff + 1] += tmp[1];
                hist[cellY][cellX].feat[setoff + 2] += tmp[2];
                hist[cellY][cellX].feat[setoff + 3] += tmp[3];
            }
            for(n=0; n<hist[cellY][cellX].numOriFull; n++)
            {
                tmp[0] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[0], 0.2);
                tmp[1] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[1], 0.2);
                tmp[2] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[2], 0.2);
                tmp[3] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[3], 0.2);
                hist[cellY][cellX].feat[hist[cellY][cellX].numOriHalf + n] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                hist[cellY][cellX].feat[setoff + 0] += tmp[0];
                hist[cellY][cellX].feat[setoff + 1] += tmp[1];
                hist[cellY][cellX].feat[setoff + 2] += tmp[2];
                hist[cellY][cellX].feat[setoff + 3] += tmp[3];
            }
        }
    }
    //cout<<"hist calculated\t"<<endl;
}

//计算HOG直方图//
void CHog::calcHist(int** orientationFullTable, //梯度方向LUT
                    int** orientationHalfTable,
                    interpItemInt** interpolationTableInt, //内差权重LUT
                    double** weightedBinTable) //权重×梯度强度LUT
{
    int x=0, y=0, n;
    int xDiff, yDiff, oriIdxFull, oriIdxHalf;
    int cellCoorX, cellCoorY, cellX, cellY, dstCellCoorX, dstCellCoorY, weight;
    int setoff = hist[0][0].numOriHalf+hist[0][0].numOriFull;
    double weightedBinValue;
    double N[4] = {0.}, tmp[4];

    //cout<<"calculating hist"<<endl;
    //遍历图像//
    for(y=0; y<imgHeight; y++)
    {
        for(x=0; x<imgWidth; x++)
        {
            xDiff = dx[y][x]; //x方向梯度
            yDiff = dy[y][x]; //y方向梯度
            oriIdxFull = orientationFullTable[yDiff+255][xDiff+255]; //量化后方向索引
            oriIdxHalf = orientationHalfTable[yDiff+255][xDiff+255];
            cellCoorY = y/cellSize + 1; //cell在hist中的y坐标
            cellCoorX = x/cellSize + 1; //cell在hist中的x坐标
            cellY = y - (cellCoorY-1)*cellSize; //图像像素在cell中的y坐标
            cellX = x - (cellCoorX-1)*cellSize; //图像像素在cell中的x坐标
            //遍历4个需要进行直方图累加的cell
            for(n=0; n<4; n++)
            {
                //计算内差cell在hist中的坐标
                dstCellCoorY = cellCoorY + interpolationTableInt[cellY][cellX].displace[n][1];
                dstCellCoorX = cellCoorX + interpolationTableInt[cellY][cellX].displace[n][0];
                weight = interpolationTableInt[cellY][cellX].weightRatio[n]; //查表获取内差权重
                weightedBinValue = weightedBinTable[weight][(yDiff+255)*NUM_DIFF + (xDiff+255)]; //查表获取内差值
                hist[dstCellCoorY][dstCellCoorX].cellBinsFull[oriIdxFull] += weightedBinValue; //计算直方图
                hist[dstCellCoorY][dstCellCoorX].cellBinsHalf[oriIdxHalf] += weightedBinValue;
            }
        }
    }

    for(cellY=0; cellY<histSizeY; cellY++)
    {
        for(cellX=0; cellX<histSizeX; cellX++)
        {
            for(n=0; n<hist[cellY][cellX].numOriHalf; n++)
                hist[cellY][cellX].normFactorHalf += hist[cellY][cellX].cellBinsHalf[n%hist[cellY][cellX].numOriFull]*hist[cellY][cellX].cellBinsHalf[n%hist[cellY][cellX].numOriFull];
        }
    }

    for(cellY=1; cellY<numCellY+1; cellY++)
    {
        for(cellX=1; cellX<numCellX+1; cellX++)
        {
            N[0] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY-1][cellX].normFactorHalf
                     + hist[cellY][cellX-1].normFactorHalf + hist[cellY-1][cellX-1].normFactorHalf);
            N[1] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY+1][cellX].normFactorHalf
                     + hist[cellY][cellX-1].normFactorHalf + hist[cellY+1][cellX-1].normFactorHalf);
            N[2] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY+1][cellX].normFactorHalf
                     + hist[cellY][cellX+1].normFactorHalf + hist[cellY+1][cellX+1].normFactorHalf);
            N[3] = sqrtf(hist[cellY][cellX].normFactorHalf + hist[cellY-1][cellX].normFactorHalf
                     + hist[cellY][cellX+1].normFactorHalf + hist[cellY-1][cellX+1].normFactorHalf);
            for(n=0; n<hist[cellY][cellX].numOriHalf; n++)
            {
                tmp[0] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[0], 0.2);
                tmp[1] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[1], 0.2);
                tmp[2] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[2], 0.2);
                tmp[3] = mmax(hist[cellY][cellX].cellBinsHalf[n]/N[3], 0.2);
                hist[cellY][cellX].feat[n] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                hist[cellY][cellX].feat[setoff + 0] += tmp[0];
                hist[cellY][cellX].feat[setoff + 1] += tmp[1];
                hist[cellY][cellX].feat[setoff + 2] += tmp[2];
                hist[cellY][cellX].feat[setoff + 3] += tmp[3];
            }
            for(n=0; n<hist[cellY][cellX].numOriFull; n++)
            {
                tmp[0] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[0], 0.2);
                tmp[1] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[1], 0.2);
                tmp[2] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[2], 0.2);
                tmp[3] = mmax(hist[cellY][cellX].cellBinsFull[n]/N[3], 0.2);
                hist[cellY][cellX].feat[hist[cellY][cellX].numOriHalf + n] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                hist[cellY][cellX].feat[setoff + 0] += tmp[0];
                hist[cellY][cellX].feat[setoff + 1] += tmp[1];
                hist[cellY][cellX].feat[setoff + 2] += tmp[2];
                hist[cellY][cellX].feat[setoff + 3] += tmp[3];
            }
        }
    }
    //cout<<"hist calculated\t"<<endl;
}

void CHog::getFeatVect(double* featVect, int x, int y, int w, int h)
{
	//cout << "getting feature vector" << endl;
    x += 1;
    y += 1;
    int dx, dy;
    if(featVect==NULL)
        featVect = new double[w*h*hist[y][x].dim];
    for(int i=0; i<h; i++)
    {
        if(y+i>=histSizeY)
            dy = histSizeY-y-1;
        else
            dy = i;
        for(int j=0; j<w; j++)
        {
            if(x+j>=histSizeX)
                dx = histSizeX-x-1;
            else
                dx = j;
            memcpy(featVect+(i*w+j)*hist[y+dy][x+dx].dim, hist[y+dy][x+dx].feat, sizeof(double)*hist[y+dy][x+dx].dim);
        }
    }
	//cout << "feature vector got" << endl;
}
void CHog::exportFeatVect(string exportPath, int x, int y, int w, int h, int l)
{
    ofstream of(exportPath.c_str(), ios::app);
    x += 1;
    y += 1;
    double* featVect = new double[w*h*hist[y][x].dim];
    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
            memcpy(featVect+(i*w+j)*hist[y+i][x+j].dim, hist[y+i][x+j].feat, sizeof(double)*hist[y+i][x+j].dim);
    }
    of<<l<<"\t";
    for(int n=0; n<w*h*hist[y][x].dim; n++)
//        of<<n+1<<":"<<featVect[n]<<"\t";
        of<<featVect[n]<<"\t";
    of<<"\n";
    of.close();
    delete[] featVect;
}

cv::Mat CHog::visualizationHalf(cv::Mat image)
{
    cout<<"visualization"<<endl;
    cv::Mat visual(cellSize*numCellY, cellSize*numCellX, CV_8UC3);
    if(image.data!=NULL)
        image.copyTo(visual);
    else
        visual.setTo(0);
    double maxBinValue = -FLT_MAX;
    int pixValue;
    cv::Point start, end, center;
    int numOri, maxIdx, maxValue;
    for(int cellY = 1; cellY<numCellY+1; cellY++)
    {
        for(int cellX = 1; cellX<numCellX+1; cellX++)
        {
            numOri = hist[cellY][cellX].numOriHalf;
            for(int b = 0; b<numOri; b++)
                maxBinValue = hist[cellY][cellX].cellBinsHalf[b]>maxBinValue ? hist[cellY][cellX].cellBinsHalf[b] : maxBinValue;
        }
    }
    for(int cellY = 1; cellY<numCellY+1; cellY++)
    {
        center.y = (cellY-1) * cellSize + cellY/2;
        for(int cellX = 1; cellX<numCellX+1; cellX++)
        {
            numOri = hist[cellY][cellX].numOriHalf;
            maxIdx = 0;
            maxValue = hist[cellY][cellX].cellBinsHalf[0]/maxBinValue*255;
            center.x = (cellX-1) * cellSize + cellX/2;

            for(int b = 0; b<numOri; b++)
            {
                pixValue = hist[cellY][cellX].cellBinsHalf[b]/maxBinValue*255;
                if(pixValue>maxValue)
                {
                    maxValue = pixValue;
                    maxIdx = b;
                }
                else
                    cv::ellipse(visual, center, cv::Size(0, cellSize/2), b*(180/numOri), 0, 360, cv::Scalar(0, pixValue, pixValue), -1);
            }
            cv::ellipse(visual, center, cv::Size(0, cellSize/2), maxIdx*(180/numOri), 0, 360, cv::Scalar(0, maxValue, maxValue), -1);
            //cv::line(visual, cv::Point((cellX-1) * cellSize, 0), cv::Point((cellX-1) * cellSize, imgHeight), cv::Scalar(255,0,0));
        }
        //cv::line(visual, cv::Point(0, (cellY-1) * cellSize), cv::Point(imgWidth, (cellY-1) * cellSize), cv::Scalar(255,0,0));
    }
    return visual;
}

cv::Mat CHog::visualizationFull(cv::Mat image)
{
    cout<<"visualization"<<endl;
    cv::Mat visual(cellSize*numCellY, cellSize*numCellX, CV_8UC3);
    if(image.data!=NULL)
        image.copyTo(visual);
    else
        visual.setTo(0);
    double maxBinValue = -FLT_MAX;
    int pixValue;
    cv::Point start, end, center;
    int numOri, maxIdx, maxValue;
    for(int cellY = 1; cellY<numCellY+1; cellY++)
    {
        for(int cellX = 1; cellX<numCellX+1; cellX++)
        {
            numOri = hist[cellY][cellX].numOriFull;
            for(int b = 0; b<numOri; b++)
                maxBinValue = hist[cellY][cellX].cellBinsFull[b]>maxBinValue ? hist[cellY][cellX].cellBinsFull[b] : maxBinValue;
        }
    }
    for(int cellY = 1; cellY<numCellY+1; cellY++)
    {
        center.y = (cellY-1) * cellSize + cellY/2;
        for(int cellX = 1; cellX<numCellX+1; cellX++)
        {
            numOri = hist[cellY][cellX].numOriFull;
            maxIdx = 0;
            maxValue = hist[cellY][cellX].cellBinsFull[0]/maxBinValue*255;
            center.x = (cellX-1) * cellSize + cellX/2;
            for(int b = 0; b<numOri; b++)
            {
                pixValue = hist[cellY][cellX].cellBinsFull[b]/maxBinValue*255;
                if(pixValue>maxValue)
                {
                    maxValue = pixValue;
                    maxIdx = b;
                }
                else
                    cv::ellipse(visual, center, cv::Size(0, cellSize/2), b*(360/numOri), 0, 360, cv::Scalar(0, pixValue, pixValue), -1);
            }
            cv::ellipse(visual, center, cv::Size(0, cellSize/2), maxIdx*(360/numOri), 0, 360, cv::Scalar(0, maxValue, maxValue), -1);
            //cv::line(visual, cv::Point((cellX-1) * cellSize, 0), cv::Point((cellX-1) * cellSize, imgHeight), cv::Scalar(255,0,0));
        }
        //cv::line(visual, cv::Point(0, (cellY-1) * cellSize), cv::Point(imgWidth, (cellY-1) * cellSize), cv::Scalar(255,0,0));
    }
    return visual;
}

cv::Mat CHog::visualizeWindow(int x, int y, int width, int height)
{
    cout<<"visualization"<<endl;
    cv::Mat visual(cellSize*height, cellSize*width, CV_8UC3);
    visual.setTo(0);
    double maxBinValue = -FLT_MAX;
    int pixValue;
    cv::Point start, end, center;
    int numOri, maxIdx, maxValue;
    x += 1;
    y += 1;
    if(x+width>numCellX+1 || y+height>numCellY+1)
    {
        cout<<"window out of image"<<endl;
        exit(0);
    }
    for(int cellY = y; cellY<y+height; cellY++)
    {
        for(int cellX = x; cellX<x+width; cellX++)
        {
            numOri = hist[cellY][cellX].numOriHalf;
            for(int b = 0; b<numOri; b++)
                maxBinValue = hist[cellY][cellX].cellBinsFull[b]>maxBinValue ? hist[cellY][cellX].cellBinsFull[b] : maxBinValue;
        }
    }
    for(int cellY = y; cellY<y+height; cellY++)
    {
        center.y = (cellY-y) * cellSize + cellY/2;
        for(int cellX = x; cellX<x+width; cellX++)
        {
            numOri = hist[cellY][cellX].numOriFull;
            maxIdx = 0;
            maxValue = hist[cellY][cellX].cellBinsFull[0]/maxBinValue*255;
            center.x = (cellX-x) * cellSize + cellX/2;
            for(int b = 0; b<numOri; b++)
            {
                pixValue = hist[cellY][cellX].cellBinsFull[b]/maxBinValue*255;
                if(pixValue>maxValue)
                {
                    maxValue = pixValue;
                    maxIdx = b;
                }
                else
                    cv::ellipse(visual, center, cv::Size(0, cellSize/2), b*(360/numOri), 0, 360, cv::Scalar(0, pixValue, pixValue), -1);
            }
            cv::ellipse(visual, center, cv::Size(0, cellSize/2), maxIdx*(360/numOri), 0, 360, cv::Scalar(0, maxValue, maxValue), -1);
            //cv::line(visual, cv::Point((cellX-x) * cellSize, 0), cv::Point((cellX-x) * cellSize, imgHeight), cv::Scalar(255,0,0));
        }
        //cv::line(visual, cv::Point(0, (cellY-1) * cellSize), cv::Point(imgWidth, (cellY-1) * cellSize), cv::Scalar(255,0,0));
    }
    return visual;
}

#endif // CHOG_H_INCLUDED
