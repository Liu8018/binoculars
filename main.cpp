#include "functions.h"
#include <opencv2/videoio.hpp>

int main()
{   
    cv::VideoCapture capture(1);
    
    //模组摄像头分辨率必须是(640*240)的一或二或四倍
    capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    
    cv::Mat frame,frame1,frame2;
    
    while(1)
    {
        //捕捉一帧
        capture >> frame;
        
        //分割左右
        frame1 = frame(cv::Range(0,frame.rows),cv::Range(0,frame.cols/2));
        frame2 = frame(cv::Range(0,frame.rows),cv::Range(frame.cols/2,frame.cols));
        
        //水平对准(模组摄像头的deltaY=-2)
        horizonAlign(frame1,frame2,-2);
        
        //在左右帧上检测特征点并计算描述子
        std::vector<cv::KeyPoint> keyPoints1;
        std::vector<cv::KeyPoint> keyPoints2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        
        cv::Ptr<cv::Feature2D> feature = cv::ORB::create(50,1,1);
        feature->detectAndCompute(frame1,cv::noArray(),keyPoints1,descriptors1);
        feature->detectAndCompute(frame2,cv::noArray(),keyPoints2,descriptors2);
        
        //水平极线约束匹配
        std::vector<cv::Point2f> leftPoints;
        std::vector<cv::Point2f> rightPoints;
        matchByHorizontalEpilines(keyPoints1,keyPoints2,descriptors1,descriptors2,leftPoints,rightPoints);
        
        //test
        cv::Mat testMatchImg;
        vizMatches(frame1,frame2,leftPoints,rightPoints,testMatchImg);
        cv::namedWindow("testMatchImg",0);
        cv::imshow("testMatchImg",testMatchImg);
        
        //显示
        //cv::namedWindow("frame1",0);
        //cv::namedWindow("frame2",0);
        //cv::imshow("frame1",frame1);
        //cv::imshow("frame2",frame2);
        
        char key = cv::waitKey();
        if(key == 'q')
            break;
        if(key == 's')
            cv::imwrite("frame.jpg",frame);
    }
    
    return 0;
}
