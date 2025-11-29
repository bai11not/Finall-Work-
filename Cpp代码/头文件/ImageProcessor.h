#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <functional>

namespace ICOOT {

// 目标函数类型枚举
enum class ObjectiveFunctionType {
    MAX_ENTROPY,
    OTSU,
    CROSS_ENTROPY
};

// 图像处理类
class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // 加载图像
    bool loadImage(const std::string& imagePath);
    
    // 转换为灰度图
    void convertToGrayscale();
    
    // 获取灰度图像
    cv::Mat getGrayscaleImage() const;
    
    // 获取原始图像
    cv::Mat getOriginalImage() const;
    
    // 应用阈值进行分割
    cv::Mat applyThresholds(const std::vector<double>& thresholds);
    
    // 生成边缘检测风格的分割结果
    cv::Mat generateEdgeStyleSegmentation(const std::vector<double>& thresholds);
    
    // 计算直方图
    std::vector<double> calculateHistogram(int bins = 256);
    
    // 保存图像
    bool saveImage(const cv::Mat& image, const std::string& filePath);

private:
    cv::Mat originalImage_;
    cv::Mat grayscaleImage_;
};

// 目标函数计算类
class ObjectiveFunctions {
public:
    // 最大熵目标函数
    static double maxEntropyObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds);
    
    // Otsu目标函数（最大化类间方差）
    static double otsuObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds);
    
    // 交叉熵目标函数
    static double crossEntropyObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds);
    
    // 根据类型获取目标函数
    static std::function<double(const std::vector<double>&, const cv::Mat&, int)> getObjectiveFunction(ObjectiveFunctionType type);
    
private:
    // 辅助函数：计算直方图和归一化
    static std::pair<std::vector<double>, std::vector<double>> calculateHistogramAndCenters(const cv::Mat& grayImage, int bins = 256);
};

} // namespace ICOOT
