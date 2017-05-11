#ifndef CTABLES_H_INCLUDED
#define CTABLES_H_INCLUDED

#include <vector>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <memory.h>
using namespace std;

const double PI = 3.1415926535897932384626433832795;
const int NUM_DIFF = 511; //微分值个数

//内差结构体//
struct interpItem
{
    int displace[4][2]; //相对于当前cell的坐标
    double weightRatio[4]; //内差权重
};

struct interpItemInt
{
    int displace[4][2]; //相对于当前cell的坐标
    int weightRatio[4]; //内差权重
};

//LUT类//
class CTables
{
public:
    CTables(int _numOriHalf, int _numOriFull, int _cellSize, bool isInt); //构造函数
    ~CTables(); //析构函数
    void saveOriTable(string outPath); //保存梯度方向lut
    void saveInteTable(string outPath); //保存内差lut
    void saveMagnTable(string outPath); //保存梯度强度lut
    void saveWeightedBinTable(string outPath); //保存内差权重×梯度强度lut
private:
    double arcTan(double x, double y); //反三角函数
    void calcOrientationTable(); //计算梯度方向lut和强度lut
    void calcInterpolationTable(); //计算内差lut
    void calcInterpolationTableInt(); //计算lut
    void calcWeightedBinTable(); //计算内差权重×梯度强度lut
public:
    int numOriHalf, numOriFull, cellSize, oriFull, oriHalf;
    int** orientationHalfTable; //梯度方向lut
    int** orientationFullTable;
    //梯度方向lut///////////////////////////
    //            x方向梯度
    //y -----------------------------....
    //方| 方向idx | 方向idx | 方向idx |....
    //向|---------------------------|....
    //梯| 方向idx | 方向idx | 方向idx |....
    //度|---------------------------|....
    //     ...      ...       ...
    //////////////////////////////////////
    double** magnitudeTable; //梯度强度lut
    //梯度强度lut//////////////////////////////
    //            x方向梯度，共511个
    //y   ------------------------------....
    //方  | 梯度强度 | 梯度强度 | 梯度强度 |....
    //向  |----------------------------|....
    //梯  | 梯度强度 | 梯度强度 | 梯度强度 |....
    //度  |----------------------------|....
    //，  | 梯度强度 | 梯度强度 | 梯度强度 |....
    //共  |  ...      ...       ...    |....
    //511 |  ...      ...       ...    |....
    //个  |  ...      ...       ...    |....
    /////////////////////////////////////////
    double** weightedBinTable; //内差权重×梯度强度lut
    //内差权重×梯度强度lut///////////////////////////////////////////
    //                  梯度强度索引,共511×511个
    //  -----------------------------------------------------....
    //内| 内差权重×梯度强度 | 内差权重×梯度强度 | 内差权重×梯度强度 |....
    //差|---------------------------------------------------|....
    //权| 内差权重×梯度强度 | 内差权重×梯度强度 | 内差权重×梯度强度 |....
    //重|---------------------------------------------------|....
    //     ...      ...       ...
    //内差权重共cellSize×cellSize个
    //////////////////////////////////////////////////////////////
    interpItem** interpolationTable; //内差lut
    //内差lut////////////////////////////////////
    //                    x方向cell中像素数
    //y -----------------------------------....
    //方| 内差结构体 | 内差结构体 | 内差结构体 |....
    //向|---------------------------------|....
    //像| 内差结构体 | 内差结构体 | 内差结构体 |....
    //素|---------------------------------|....
    //数   ...      ...       ...
    //
    /////////////////////////////////////////////
    interpItemInt** interpolationTableInt; //内差lut
};

CTables::CTables(int _numOriHalf, int _numOriFull, int _cellSize, bool isInt)
{
    numOriHalf = _numOriHalf;
    numOriFull = _numOriFull;
    cellSize = _cellSize;
    oriFull = 360;
    oriHalf = 180;
    calcOrientationTable();
    if(isInt)
    {
        calcInterpolationTableInt();
        interpolationTable = NULL;
    }
    else
    {
        calcInterpolationTable();
        interpolationTableInt = NULL;
    }
    calcWeightedBinTable();
}

CTables::~CTables()
{
    for(int y=0; y<NUM_DIFF; y++)
    {
        delete[] orientationFullTable[y];
        delete[] orientationHalfTable[y];
        delete[] magnitudeTable[y];
    }
    delete[] orientationFullTable;
    delete[] orientationHalfTable;
    delete[] magnitudeTable;

    if(interpolationTable!=NULL)
    {
        for(int y=0; y<cellSize; y++)
            delete[] interpolationTable[y];
        delete[] interpolationTable;
    }
    if(interpolationTableInt!=NULL)
    {
        for(int y=0; y<cellSize; y++)
            delete[] interpolationTableInt[y];
        delete[] interpolationTableInt;
    }

    for(int y=0; y<cellSize*cellSize; y++)
    {
        delete[] weightedBinTable[y];
    }
    delete[] weightedBinTable;
}

double CTables::arcTan(double x, double y)
{
    double radian = atan2(y, x);
    double degree = radian/PI*180;
    return degree;
}

void CTables::calcOrientationTable()
{
    orientationFullTable = new int*[NUM_DIFF];
    orientationHalfTable = new int*[NUM_DIFF];
    magnitudeTable = new double*[NUM_DIFF];
    double intervalFull = oriFull/numOriFull;
    double intervalHalf = oriHalf/numOriHalf;
    for(int y=-255; y<=255; y++)
    {
        orientationFullTable[y+255] = new int[NUM_DIFF];
        orientationHalfTable[y+255] = new int[NUM_DIFF];
        magnitudeTable[y+255] = new double[NUM_DIFF];
        for(int x=-255; x<=255; x++)
        {
            orientationFullTable[y+255][x+255] = (((int)arcTan(x, y)+360)%oriFull)/intervalFull;
            orientationHalfTable[y+255][x+255] = (((int)arcTan(x, y)+360)%oriHalf)/intervalHalf;
            magnitudeTable[y+255][x+255] = sqrtf(x*x + y*y);
        }
    }
    cout<<"ori and magn table created"<<endl;
}

void CTables::calcInterpolationTableInt()
{
    interpolationTableInt = new interpItemInt*[cellSize];
    for(int y=-cellSize/2; y<cellSize/2; y++)
    {
        interpolationTableInt[y+cellSize/2] = new interpItemInt[cellSize];
        for(int x=-cellSize/2; x<cellSize/2; x++)
        {
            if(y<0 && x <0)
            {
                int nearWeightX = cellSize + x + 1;
                int farWeightX = cellSize - nearWeightX;
                int nearWeightY = cellSize + y + 1;
                int farWeightY = cellSize - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][1] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][0] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][0] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][1] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y<0 && x>=0)
            {
                int nearWeightX = cellSize - x;
                int farWeightX = cellSize - nearWeightX;
                int nearWeightY = cellSize + y + 1;
                int farWeightY = cellSize - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][1] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][0] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][0] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][1] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y>=0 && x>=0)
            {
                int nearWeightX = cellSize - x;
                int farWeightX = cellSize - nearWeightX;
                int nearWeightY = cellSize - y;
                int farWeightY = cellSize - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][1] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][0] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][0] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][1] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y>=0 && x<0)
            {
                int nearWeightX = cellSize + x + 1;
                int farWeightX = cellSize - nearWeightX;
                int nearWeightY = cellSize - y;
                int farWeightY = cellSize - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[1][1] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][0] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][0] = -1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].displace[3][1] = 1;
                interpolationTableInt[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else
            {
                cout<<"err"<<endl;
                exit(0);
            }
        }
    }
    cout<<"interTable created"<<endl;
}

void CTables::calcInterpolationTable()
{
    interpolationTable = new interpItem*[cellSize];
    for(int y=-cellSize/2; y<cellSize/2; y++)
    {
        interpolationTable[y+cellSize/2] = new interpItem[cellSize];
        for(int x=-cellSize/2; x<cellSize/2; x++)
        {
            if(y<0 && x <0)
            {
                double nearWeightX = (double)(cellSize + x + 1)/cellSize;
                double farWeightX = 1.0 - nearWeightX;
                double nearWeightY = (double)(cellSize + y + 1)/cellSize;
                double farWeightY = 1.0 - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][1] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][0] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][0] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][1] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y<0 && x>=0)
            {
                double nearWeightX = (double)(cellSize - x)/cellSize;
                double farWeightX = 1.0 - nearWeightX;
                double nearWeightY = (double)(cellSize + y + 1)/cellSize;
                double farWeightY = 1.0 - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][1] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][0] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][0] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][1] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y>=0 && x>=0)
            {
                double nearWeightX = (double)(cellSize - x)/cellSize;
                double farWeightX = 1.0 - nearWeightX;
                double nearWeightY = (double)(cellSize - y)/cellSize;
                double farWeightY = 1.0 - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][1] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][0] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][0] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][1] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else if(y>=0 && x<0)
            {
                double nearWeightX = (double)(cellSize + x + 1)/cellSize;
                double farWeightX = 1.0 - nearWeightX;
                double nearWeightY = (double)(cellSize - y)/cellSize;
                double farWeightY = 1.0 - nearWeightY;
                //cout<<nearWeightX<<"\t"<<nearWeightY<<endl;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[0][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[0] = nearWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][0] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[1][1] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[1] = nearWeightX*farWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][0] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[2][1] = 0;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[2] = farWeightX*nearWeightY;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][0] = -1;
                interpolationTable[y+cellSize/2][x+cellSize/2].displace[3][1] = 1;
                interpolationTable[y+cellSize/2][x+cellSize/2].weightRatio[3] = farWeightX*farWeightY;
            }
            else
            {
                cout<<"err"<<endl;
                exit(0);
            }
        }
    }
    cout<<"interTable created"<<endl;
}

void CTables::calcWeightedBinTable()
{
    int cellSize2 = cellSize*cellSize;
    weightedBinTable = new double*[cellSize2 + 1];
    for(int w=0; w<cellSize2+1; w++)
    {
        weightedBinTable[w] = new double[NUM_DIFF*NUM_DIFF];
        memset(weightedBinTable[w], 0, NUM_DIFF*NUM_DIFF*sizeof(double));
    }

    for(int w=0; w<cellSize2+1; w++)
    {
        for(int y=-255; y<=255; y++)
        {
            for(int x=-255; x<=255; x++)
            {
                weightedBinTable[w][(y+255)*NUM_DIFF+(x+255)] = magnitudeTable[y+255][x+255]*w/cellSize2;
                //cout<<magnitudeTable[y+255][x+255]<<"\t"<<magnitudeTable[y+255][x+255]*w<<"\t"<<magnitudeTable[y+255][x+255]*w/cellSize2<<endl;
            }
        }
    }
    cout<<"WeightedBinTable created"<<endl;
}

void CTables::saveOriTable(string outPath)
{
    ofstream gof(outPath.c_str());
    gof<<"full orientation table"<<endl;
    for(int y=0; y<NUM_DIFF; y++)
    {
        for(int x=0; x<NUM_DIFF; x++)
            gof<<orientationFullTable[y][x]<<"\t";
        gof<<endl;
    }
    gof<<"half orientation table"<<endl;
    for(int y=0; y<NUM_DIFF; y++)
    {
        for(int x=0; x<NUM_DIFF; x++)
            gof<<orientationHalfTable[y][x]<<"\t";
        gof<<endl;
    }
    gof.close();
    cout<<"gradTable stored"<<endl;
}

void CTables::saveMagnTable(string outPath)
{
    ofstream mof(outPath.c_str());
    for(int y=0; y<NUM_DIFF; y++)
    {
        for(int x=0; x<NUM_DIFF; x++)
            mof<<magnitudeTable[y][x]<<"\t";
        mof<<endl;
    }
    mof.close();
    cout<<"magnTable stored"<<endl;
}

void CTables::saveWeightedBinTable(string outPath)
{
    ofstream mof(outPath.c_str());
    for(int y=0; y<cellSize*cellSize; y++)
    {
        for(int x=0; x<NUM_DIFF*NUM_DIFF; x++)
            mof<<weightedBinTable[y][x]<<"\t";
        mof<<endl;
    }
    mof.close();
    cout<<"weightedBinTable stored"<<endl;
}

void CTables::saveInteTable(string outPath)
{
    ofstream iof(outPath.c_str());
    if(interpolationTable!=NULL)
    {
        for(int y=0; y<cellSize; y++)
        {
            for(int x=0; x<cellSize; x++)
            {
                for(int i=0; i<4; i++)
                {
                    iof<<"xy:"<<x<<"\t"<<y<<"\tdisplace:"<<interpolationTable[y][x].displace[i][0]<<"\t"<<interpolationTable[y][x].displace[i][1]
                        <<"\tweightRatio:"<<interpolationTable[y][x].weightRatio[i]<<"\n";
                }
            }
            iof<<endl;
        }
    }
    if(interpolationTableInt!=NULL)
    {
        for(int y=0; y<cellSize; y++)
        {
            for(int x=0; x<cellSize; x++)
            {
                for(int i=0; i<4; i++)
                {
                    iof<<"xy:"<<x<<"\t"<<y<<"\tdisplace:"<<interpolationTableInt[y][x].displace[i][0]<<"\t"<<interpolationTableInt[y][x].displace[i][1]
                        <<"\tweightRatio:"<<interpolationTableInt[y][x].weightRatio[i]<<"\n";
                }
            }
            iof<<endl;
        }
    }
    iof.close();
    cout<<"interTable stored"<<endl;
}

#endif // CTABLES_H_INCLUDED
