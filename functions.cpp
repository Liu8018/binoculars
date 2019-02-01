#include "functions.h"

void findFMAndShowEpilines(const std::vector<cv::Point> &pts1, const std::vector<cv::Point> &pts2,cv::Mat &img1, cv::Mat &img2)
{
    //在两张图片上绘制特征点
    for(int i=0;i<pts1.size();i++)
        cv::circle(img1,pts1[i],4,cv::Scalar(255,255,0),-1);
    for(int i=0;i<pts2.size();i++)
        cv::circle(img2,pts2[i],4,cv::Scalar(255,255,0),-1);
    
    //计算基础矩阵
    cv::Mat fundamentalMat = cv::findFundamentalMat(pts1,pts2);
    
    //绘制极线
    std::vector<cv::Vec3f> epilines1;
    cv::computeCorrespondEpilines(pts1,1,fundamentalMat,epilines1);
    for(int i=0;i<epilines1.size();i++)
    {
        cv::line(img2,cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
                cv::Point(img2.cols,-(epilines1[i][2]+epilines1[i][0]*img2.cols)/epilines1[i][1]),
                cv::Scalar(255,255,255),2);
    }
    
    std::vector<cv::Vec3f> epilines2;
    cv::computeCorrespondEpilines(pts2,2,fundamentalMat,epilines2);
    for(int i=0;i<epilines2.size();i++)
    {
        cv::line(img1,cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
                cv::Point(img1.cols,-(epilines2[i][2]+epilines2[i][0]*img1.cols)/epilines2[i][1]),
                cv::Scalar(255,255,255),2);
    }
}

void horizonAlign(cv::Mat &img1, cv::Mat &img2, int deltaY)
{
    if(img1.size() != img2.size())
    {
        std::cout<<"[horizonAlign] Error: Images' size not match!"<<std::endl;
        return;
    }
    
    int rows = img1.rows;
    int cols = img1.cols;
    
    if(deltaY > 0)
    {
        img2 = img2(cv::Range(deltaY,rows),cv::Range(0,cols));
        img1 = img1(cv::Range(0,rows-deltaY),cv::Range(0,cols));
    }
    if(deltaY < 0)
    {
        deltaY = -deltaY;
        img1 = img1(cv::Range(deltaY,rows),cv::Range(0,cols));
        img2 = img2(cv::Range(0,rows-deltaY),cv::Range(0,cols));
    }
}

void addLine(cv::Mat &dstMat, const cv::Mat &srcMat, int index)
{
    cv::Mat srcROILine = srcMat(cv::Range(index,index+1),cv::Range(0,srcMat.cols));
    
    if(dstMat.empty())
        srcROILine.copyTo(dstMat);
    else
    {
        cv::Mat dstMat2(dstMat.rows+1,dstMat.cols,dstMat.type());
        
        cv::Mat ROI1 = dstMat2(cv::Range(0,dstMat.rows),cv::Range(0,dstMat2.cols));
        cv::Mat ROI2 = dstMat2(cv::Range(dstMat.rows,dstMat2.rows),cv::Range(0,dstMat2.cols));
        
        dstMat.copyTo(ROI1);
        srcROILine.copyTo(ROI2);
        dstMat2.copyTo(dstMat);
    }
}

void matchByHorizontalEpilines(cv::Size imgSize, int split,
                               const std::vector<cv::KeyPoint> &keyPoints1, const std::vector<cv::KeyPoint> &keyPoints2, 
                               const cv::Mat &descriptors1, const cv::Mat &descriptors2, 
                               std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints)
{
    int h = imgSize.height;
    
    std::vector<cv::KeyPoint> box_keyPoints1[split];
    std::vector<cv::KeyPoint> box_keyPoints2[split];
    cv::Mat box_descriptors1[split];
    cv::Mat box_descriptors2[split];
        
    for(int i=0;i<keyPoints1.size();i++)
    {
        int y = keyPoints1[i].pt.y;
        box_keyPoints1[y*split/h].push_back(keyPoints1[i]);
        addLine(box_descriptors1[y*split/h],descriptors1,i);
    }
    for(int i=0;i<keyPoints2.size();i++)
    {
        int y = keyPoints2[i].pt.y;
        box_keyPoints2[y*split/h].push_back(keyPoints2[i]);
        addLine(box_descriptors2[y*split/h],descriptors2,i);
    }
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches[split];
    for(int i=0;i<split;i++)
    {
        if(box_descriptors1[i].empty() || box_descriptors2[i].empty())
            continue;
            
        matcher.match(box_descriptors1[i],box_descriptors2[i],matches[i]);
        
        for(int j=0;j<matches[i].size();j++)
        {
            leftPoints.push_back(box_keyPoints1[i][matches[i][j].queryIdx].pt);
            rightPoints.push_back(box_keyPoints2[i][matches[i][j].trainIdx].pt);
        }
    }
}

void matchByHorizontalEpilines(const std::vector<cv::KeyPoint> &keyPoints1, const std::vector<cv::KeyPoint> &keyPoints2, 
                               const cv::Mat &descriptors1, const cv::Mat &descriptors2, 
                               std::vector<cv::Point2f> &leftPoints, std::vector<cv::Point2f> &rightPoints)
{
    //匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches);
    
    //筛选
    for(int i=0;i<matches.size();i++)
    {
        cv::Point2f p1 = keyPoints1[matches[i].queryIdx].pt;
        cv::Point2f p2 = keyPoints2[matches[i].trainIdx].pt;
        
        //y相等，x1大于x2
        if(std::abs(p1.y-p2.y)<2 && p1.x>p2.x)
        {
            leftPoints.push_back(p1);
            rightPoints.push_back(p2);
        }
    }
}

void vizMatches(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &vizImg)
{
    std::vector<cv::KeyPoint> keyPoints1;
    std::vector<cv::KeyPoint> keyPoints2;
    cv::KeyPoint::convert(pts1,keyPoints1);
    cv::KeyPoint::convert(pts2,keyPoints2);
    
    std::vector<cv::DMatch> matches(pts1.size());
    for(int i=0;i<pts1.size();i++)
    {
        matches[i].queryIdx = i;
        matches[i].trainIdx = i;
    }
    
    cv::drawMatches(img1,keyPoints1,img2,keyPoints2,matches,vizImg);
    
    for(int i=0;i<pts1.size();i++)
        cv::putText(vizImg,std::to_string(pts1[i].x - pts2[i].x).substr(0,3),pts1[i],0,0.3,cv::Scalar(255,0,0));
}
