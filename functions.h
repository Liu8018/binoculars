#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//根据匹配好的点计算基础矩阵并绘制极线
void findFMAndShowEpilines(const std::vector<cv::Point> &pts1, 
                           const std::vector<cv::Point> &pts2, 
                           cv::Mat &img1, cv::Mat &img2);

//水平对齐两张图像
void horizonAlign(cv::Mat &img1, cv::Mat &img2, int deltaY);

//根据水平极线约束匹配特征点。先划分水平区间，再逐区间匹配
void matchByHorizontalEpilines(cv::Size imgSize, 
                               int split,
                               const std::vector<cv::KeyPoint> &keyPoints1, 
                               const std::vector<cv::KeyPoint> &keyPoints2,
                               const cv::Mat &descriptors1, 
                               const cv::Mat &descriptors2,
                               std::vector<cv::Point2f> &leftPoints, 
                               std::vector<cv::Point2f> &rightPoints);
//根据水平极线约束匹配特征点。先对整张图像匹配，再筛选
void matchByHorizontalEpilines(const std::vector<cv::KeyPoint> &keyPoints1, 
                               const std::vector<cv::KeyPoint> &keyPoints2,
                               const cv::Mat &descriptors1, 
                               const cv::Mat &descriptors2,
                               std::vector<cv::Point2f> &leftPoints, 
                               std::vector<cv::Point2f> &rightPoints);

//匹配点可视化
void vizMatches(const cv::Mat &img1, 
                const cv::Mat &img2,
                const std::vector<cv::Point2f> &pts1, 
                const std::vector<cv::Point2f> &pts2,
                cv::Mat &vizImg);
