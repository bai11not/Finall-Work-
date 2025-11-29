#pragma once
#include "ImageProcessor.h"
#include "ICOOTOptimizer.h"
#include <string>
#include <vector>
#include <map>

namespace ICOOT {

// 批量处理器类，用于处理多图像数据集
class BatchProcessor {
public:
    BatchProcessor();
    ~BatchProcessor();
    
    // 设置参数
    void setParameters(int numThresholds, ObjectiveFunctionType objFuncType,
                      int populationSize = 50, int maxIterations = 100,
                      bool outputEdgeStyle = true, bool showProgress = true);
    
    // 处理单个图像
    bool processSingleImage(const std::string& imagePath, const std::string& resultsPath);
    
    // 批量处理文件夹中的图像
    bool processDataset(const std::string& datasetPath, const std::string& resultsPath);
    
    // 生成汇总报告
    bool generateSummaryReport(const std::string& reportPath);
    
    // 获取处理统计信息
    std::map<std::string, std::string> getStatistics() const;
    
    // 设置是否保存中间数据
    void setSaveIntermediateData(bool save);
    
    // 重置统计信息
    void resetStatistics();

private:
    // 处理参数
    int numThresholds_;
    ObjectiveFunctionType objFuncType_;
    int populationSize_;
    int maxIterations_;
    bool outputEdgeStyle_;
    bool showProgress_;
    bool saveIntermediateData_;
    
    // 统计信息
    int totalImages_;
    int processedImages_;
    int failedImages_;
    double totalProcessingTime_;
    std::map<std::string, std::vector<double>> results_; // 存储每个图像的结果
    
    // 获取文件夹中的所有图像文件
    std::vector<std::string> getImageFiles(const std::string& directory);
    
    // 提取图像文件名（不含扩展名）
    std::string extractFileName(const std::string& filePath);
    
    // 确保结果目录存在
    bool ensureDirectory(const std::string& directoryPath);
    
    // 保存结果信息到文本文件
    bool saveResultInfo(const std::string& imageName, const std::string& resultsPath,
                       const std::vector<double>& thresholds, double fitness,
                       const std::vector<double>& convergenceCurve);
    
    // 执行ICOOT分割
    bool performICOOTSegmentation(const cv::Mat& grayImage, const std::string& imageName,
                                const std::string& resultsPath, std::vector<double>& outThresholds,
                                double& outFitness, std::vector<double>& outConvergenceCurve);
};

} // namespace ICOOT
