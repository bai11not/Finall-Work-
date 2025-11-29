#include "../include/BatchProcessor.h"
#include "../include/ICOOTOptimizer.h"
#include "../include/ImageProcessor.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <regex>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace ICOOT {

BatchProcessor::BatchProcessor() {
    // 设置默认参数
    setParameters(3, ObjectiveFunctionType::MAX_ENTROPY);
    saveIntermediateData_ = true;
    resetStatistics();
    
    // 设置控制台编码为UTF-8，确保中文正常显示
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
}

BatchProcessor::~BatchProcessor() {
}

void BatchProcessor::resetStatistics() {
    totalImages_ = 0;
    processedImages_ = 0;
    failedImages_ = 0;
    totalProcessingTime_ = 0.0;
    results_.clear();
}

void BatchProcessor::setParameters(int numThresholds, ObjectiveFunctionType objFuncType,
                                  int populationSize, int maxIterations,
                                  bool outputEdgeStyle, bool showProgress) {
    numThresholds_ = numThresholds;
    objFuncType_ = objFuncType;
    populationSize_ = populationSize;
    maxIterations_ = maxIterations;
    outputEdgeStyle_ = outputEdgeStyle;
    showProgress_ = showProgress;
}

bool BatchProcessor::processSingleImage(const std::string& imagePath, const std::string& resultsPath) {
    // 确保结果目录存在
    if (!ensureDirectory(resultsPath)) {
        std::cerr << "Failed to create results directory: " << resultsPath << std::endl;
        return false;
    }
    
    // 提取文件名
    std::string imageName = extractFileName(imagePath);
    
    // 加载图像
    ImageProcessor imageProcessor;
    if (!imageProcessor.loadImage(imagePath)) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        failedImages_++;
        return false;
    }
    
    // 获取灰度图像
    cv::Mat grayImage = imageProcessor.getGrayscaleImage();
    
    // 记录开始时间
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 执行ICOOT分割
    std::vector<double> thresholds;
    double fitness;
    std::vector<double> convergenceCurve;
    
    if (!performICOOTSegmentation(grayImage, imageName, resultsPath, thresholds, fitness, convergenceCurve)) {
        std::cerr << "ICOOT segmentation failed for image: " << imageName << std::endl;
        failedImages_++;
        return false;
    }
    
    // 计算处理时间
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();
    totalProcessingTime_ += processingTime;
    
    // 应用阈值分割
    cv::Mat segmentedImage;
    if (outputEdgeStyle_) {
        segmentedImage = imageProcessor.generateEdgeStyleSegmentation(thresholds);
    } else {
        segmentedImage = imageProcessor.applyThresholds(thresholds);
    }
    
    // 获取原始图像
    cv::Mat originalImage = imageProcessor.getOriginalImage();
    
    // 显示原始图像和分割结果
    try {
        std::cout << "\n[显示图像] 准备显示原始图像和分割结果..." << std::endl;
        
        // 确保图像格式适合显示
        cv::Mat displaySegmented;
        if (segmentedImage.type() == CV_64F) {
            cv::normalize(segmentedImage, displaySegmented, 0, 1, cv::NORM_MINMAX);
            displaySegmented.convertTo(displaySegmented, CV_8U, 255.0);
        } else if (segmentedImage.type() == CV_32F) {
            cv::normalize(segmentedImage, displaySegmented, 0, 1, cv::NORM_MINMAX);
            displaySegmented.convertTo(displaySegmented, CV_8U, 255.0);
        } else {
            displaySegmented = segmentedImage.clone();
        }
        
        // 如果是单通道图像，转换为三通道以便显示彩色文本
        if (displaySegmented.channels() == 1) {
            cv::cvtColor(displaySegmented, displaySegmented, cv::COLOR_GRAY2BGR);
        }
        
        // 创建显示窗口
        cv::namedWindow("原图", cv::WINDOW_NORMAL);
        cv::namedWindow("分割结果", cv::WINDOW_NORMAL);
        
        // 调整窗口大小
        cv::resizeWindow("原图", 600, 400);
        cv::resizeWindow("分割结果", 600, 400);
        
        // 在分割结果图像上添加文本信息
        std::string text = "阈值: ";
        for (size_t i = 0; i < thresholds.size(); i++) {
            text += std::to_string(thresholds[i]);
            if (i < thresholds.size() - 1) text += ", ";
        }
        
        std::string fitnessText = "适应度: " + std::to_string(fitness);
        
        // 添加文本到分割结果图像
        cv::putText(displaySegmented, text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(displaySegmented, fitnessText, cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 显示图像
        cv::imshow("原图", originalImage);
        cv::imshow("分割结果", displaySegmented);
        
        std::cout << "[显示图像] 图像已显示。按任意键继续..." << std::endl;
        
        // 等待用户按键
        cv::waitKey(0);
        
        // 关闭窗口
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "[警告] 无法显示图像: " << e.what() << std::endl;
        std::cerr << "这可能是因为环境不支持图形界面显示" << std::endl;
    }
    
    // 保存分割结果
    std::string outputPath = resultsPath + "\\" + imageName + "_segmented.png";
    std::cout << "Saving segmented image to: " << outputPath << std::endl;
    
    // 检查图像是否为空
    if (segmentedImage.empty()) {
        std::cerr << "Error: Segmented image is empty!" << std::endl;
        return false;
    }
    
    // 确保图像深度正确
    cv::Mat saveImage;
    if (segmentedImage.type() == CV_64F) {
        // 先归一化到[0,1]范围
        cv::normalize(segmentedImage, saveImage, 0, 1, cv::NORM_MINMAX);
        // 转换为8位无符号整型
        saveImage.convertTo(saveImage, CV_8U, 255.0);
    } else {
        saveImage = segmentedImage.clone();
    }
    
    // 直接使用cv::imwrite保存图像
    if (!cv::imwrite(outputPath, saveImage)) {
        std::cerr << "Error: Failed to save image using cv::imwrite to " << outputPath << std::endl;
        std::cerr << "  - Check if output directory exists: " << resultsPath << std::endl;
        std::cerr << "  - Check if OpenCV is properly linked" << std::endl;
        std::cerr << "  - Check if image is valid: rows=" << saveImage.rows << ", cols=" << saveImage.cols << ", type=" << saveImage.type() << std::endl;
        
        // 尝试保存到当前目录作为备选
        std::string altOutputPath = imageName + "_segmented.png";
        std::cout << "Attempting to save to alternative path: " << altOutputPath << std::endl;
        if (cv::imwrite(altOutputPath, saveImage)) {
            std::cout << "Successfully saved to alternative path: " << altOutputPath << std::endl;
        } else {
            std::cerr << "Failed to save to alternative path as well!" << std::endl;
            return false;
        }
    } else {
        std::cout << "Successfully saved segmented image to: " << outputPath << std::endl;
    }
    
    // 保存原始图像的副本（可选，用于调试）
    std::string originalOutputPath = resultsPath + "\\" + imageName + "_original.png";
    if (!imageProcessor.saveImage(imageProcessor.getOriginalImage(), originalOutputPath)) {
        std::cout << "Note: Failed to save original image copy (optional)" << std::endl;
    }
    
    // 保存结果信息
    if (saveIntermediateData_) {
        if (!saveResultInfo(imageName, resultsPath, thresholds, fitness, convergenceCurve)) {
            std::cerr << "Failed to save result info for image: " << imageName << std::endl;
        }
    }
    
    // 更新统计信息
    processedImages_++;
    
    // 存储结果
    results_[imageName + "_thresholds"] = thresholds;
    results_[imageName + "_fitness"].push_back(fitness);
    results_[imageName + "_time"].push_back(processingTime);
    
    std::cout << "Processed image: " << imageName << " in " << processingTime << " seconds" << std::endl;
    
    return true;
}

bool BatchProcessor::processDataset(const std::string& datasetPath, const std::string& resultsPath) {
    resetStatistics();
    
    // 获取所有图像文件
    std::vector<std::string> imageFiles = getImageFiles(datasetPath);
    totalImages_ = static_cast<int>(imageFiles.size());
    
    if (totalImages_ == 0) {
        std::cerr << "No image files found in directory: " << datasetPath << std::endl;
        return false;
    }
    
    std::cout << "Found " << totalImages_ << " image files to process." << std::endl;
    
    // 确保结果目录存在
    if (!ensureDirectory(resultsPath)) {
        std::cerr << "Failed to create results directory: " << resultsPath << std::endl;
        return false;
    }
    
    // 创建错误日志文件
    std::string errorLogPath = resultsPath + "\\error_log.txt";
    std::ofstream errorLog(errorLogPath);
    
    // 处理每个图像
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        const std::string& imageFile = imageFiles[i];
        std::cout << "Processing image " << (i + 1) << "/" << totalImages_ << ": " << imageFile << std::endl;
        
        try {
            if (!processSingleImage(imageFile, resultsPath)) {
                if (errorLog.is_open()) {
                    errorLog << "Failed to process: " << imageFile << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception processing image " << imageFile << ": " << e.what() << std::endl;
            if (errorLog.is_open()) {
                errorLog << "Exception processing: " << imageFile << ": " << e.what() << std::endl;
            }
            failedImages_++;
        }
    }
    
    // 关闭错误日志
    if (errorLog.is_open()) {
        errorLog.close();
    }
    
    // 生成汇总报告
    generateSummaryReport(resultsPath + "\\summary_report.txt");
    
    // 输出处理统计
    std::cout << "\nProcessing Summary:" << std::endl;
    std::cout << "Total images: " << totalImages_ << std::endl;
    std::cout << "Successfully processed: " << processedImages_ << std::endl;
    std::cout << "Failed: " << failedImages_ << std::endl;
    std::cout << "Average processing time per image: " 
              << (processedImages_ > 0 ? totalProcessingTime_ / processedImages_ : 0.0) << " seconds" << std::endl;
    
    return processedImages_ > 0;
}

bool BatchProcessor::generateSummaryReport(const std::string& reportPath) {
    std::ofstream report(reportPath);
    if (!report.is_open()) {
        std::cerr << "Failed to create summary report: " << reportPath << std::endl;
        return false;
    }
    
    // 写入报告标题
    report << "ICOOT Segmentation Summary Report" << std::endl;
    report << "==================================" << std::endl;
    report << std::endl;
    
    // 写入参数信息
    report << "Processing Parameters:" << std::endl;
    report << "- Number of thresholds: " << numThresholds_ << std::endl;
    
    std::string objFuncName;
    switch (objFuncType_) {
        case ObjectiveFunctionType::MAX_ENTROPY:
            objFuncName = "Max Entropy";
            break;
        case ObjectiveFunctionType::OTSU:
            objFuncName = "Otsu";
            break;
        case ObjectiveFunctionType::CROSS_ENTROPY:
            objFuncName = "Cross Entropy";
            break;
        default:
            objFuncName = "Unknown";
    }
    
    report << "- Objective function: " << objFuncName << std::endl;
    report << "- Population size: " << populationSize_ << std::endl;
    report << "- Max iterations: " << maxIterations_ << std::endl;
    report << "- Output style: " << (outputEdgeStyle_ ? "Edge Detection" : "Multi-region Segmentation") << std::endl;
    report << std::endl;
    
    // 写入统计信息
    report << "Processing Statistics:" << std::endl;
    report << "- Total images: " << totalImages_ << std::endl;
    report << "- Successfully processed: " << processedImages_ << std::endl;
    report << "- Failed: " << failedImages_ << std::endl;
    report << "- Total processing time: " << totalProcessingTime_ << " seconds" << std::endl;
    report << "- Average processing time per image: " 
           << (processedImages_ > 0 ? totalProcessingTime_ / processedImages_ : 0.0) << " seconds" << std::endl;
    report << std::endl;
    
    // 写入每个图像的结果摘要
    report << "Individual Image Results:" << std::endl;
    report << "--------------------------" << std::endl;
    
    // 收集所有唯一的图像名称
    std::set<std::string> imageNames;
    for (const auto& [key, _] : results_) {
        if (key.find("_thresholds") != std::string::npos) {
            std::string imageName = key.substr(0, key.find("_thresholds"));
            imageNames.insert(imageName);
        }
    }
    
    for (const auto& imageName : imageNames) {
        report << "\nImage: " << imageName << std::endl;
        
        // 写入阈值
        auto thresholdsIter = results_.find(imageName + "_thresholds");
        if (thresholdsIter != results_.end()) {
            report << "Thresholds: [";
            for (size_t i = 0; i < thresholdsIter->second.size(); ++i) {
                report << thresholdsIter->second[i];
                if (i < thresholdsIter->second.size() - 1) {
                    report << ", ";
                }
            }
            report << "]" << std::endl;
        }
        
        // 写入适应度值
        auto fitnessIter = results_.find(imageName + "_fitness");
        if (fitnessIter != results_.end() && !fitnessIter->second.empty()) {
            report << "Fitness: " << fitnessIter->second[0] << std::endl;
        }
        
        // 写入处理时间
        auto timeIter = results_.find(imageName + "_time");
        if (timeIter != results_.end() && !timeIter->second.empty()) {
            report << "Processing time: " << timeIter->second[0] << " seconds" << std::endl;
        }
    }
    
    report.close();
    std::cout << "Summary report generated: " << reportPath << std::endl;
    return true;
}

std::map<std::string, std::string> BatchProcessor::getStatistics() const {
    std::map<std::string, std::string> stats;
    stats["total_images"] = std::to_string(totalImages_);
    stats["processed_images"] = std::to_string(processedImages_);
    stats["failed_images"] = std::to_string(failedImages_);
    stats["total_time"] = std::to_string(totalProcessingTime_);
    stats["avg_time"] = processedImages_ > 0 ? 
                        std::to_string(totalProcessingTime_ / processedImages_) : "0.0";
    return stats;
}

void BatchProcessor::setSaveIntermediateData(bool save) {
    saveIntermediateData_ = save;
}

std::vector<std::string> BatchProcessor::getImageFiles(const std::string& directory) {
    std::vector<std::string> imageFiles;
    
    try {
        // 支持的图像扩展名
        std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
        
        // 检查目录是否存在
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Directory does not exist: " << directory << std::endl;
            return imageFiles;
        }
        
        // 遍历目录中的所有文件
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (fs::is_regular_file(entry)) {
                std::string path = entry.path().string();
                std::string ext = fs::path(path).extension().string();
                
                // 转换扩展名为小写进行比较
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // 检查是否为支持的图像格式
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    imageFiles.push_back(path);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }
    
    return imageFiles;
}

std::string BatchProcessor::extractFileName(const std::string& filePath) {
    fs::path path(filePath);
    return path.stem().string();
}

bool BatchProcessor::ensureDirectory(const std::string& directoryPath) {
    try {
        if (!fs::exists(directoryPath)) {
            return fs::create_directories(directoryPath);
        }
        return fs::is_directory(directoryPath);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return false;
    }
}

bool BatchProcessor::saveResultInfo(const std::string& imageName, const std::string& resultsPath,
                                   const std::vector<double>& thresholds, double fitness,
                                   const std::vector<double>& convergenceCurve) {
    std::string infoFilePath = resultsPath + "\\" + imageName + "_info.txt";
    std::ofstream infoFile(infoFilePath);
    
    if (!infoFile.is_open()) {
        std::cerr << "Failed to create info file: " << infoFilePath << std::endl;
        return false;
    }
    
    infoFile << "ICOOT Segmentation Results" << std::endl;
    infoFile << "Image: " << imageName << std::endl;
    infoFile << "Number of thresholds: " << thresholds.size() << std::endl;
    infoFile << "\nThresholds: " << std::endl;
    
    for (size_t i = 0; i < thresholds.size(); ++i) {
        infoFile << "T" << (i + 1) << ": " << thresholds[i] << std::endl;
    }
    
    infoFile << "\nBest Fitness Value: " << fitness << std::endl;
    infoFile << "\nConvergence Curve (iterations): " << std::endl;
    
    // 只保存前100个迭代值或全部（如果少于100）
    size_t maxIterToSave = std::min<size_t>(100, convergenceCurve.size());
    for (size_t i = 0; i < maxIterToSave; ++i) {
        infoFile << "Iter " << (i + 1) << ": " << convergenceCurve[i] << std::endl;
    }
    
    if (convergenceCurve.size() > maxIterToSave) {
        infoFile << "... " << (convergenceCurve.size() - maxIterToSave) << " more iterations omitted ..." << std::endl;
    }
    
    infoFile.close();
    return true;
}

bool BatchProcessor::performICOOTSegmentation(const cv::Mat& grayImage, const std::string& imageName,
                                            const std::string& resultsPath, std::vector<double>& outThresholds,
                                            double& outFitness, std::vector<double>& outConvergenceCurve) {
    try {
        // 设置优化参数
        int dim = numThresholds_;
        std::vector<double> lb(dim, 0.01); // 阈值下界，避免0
        std::vector<double> ub(dim, 0.99); // 阈值上界，避免1
        
        // 获取目标函数
        auto objFunc = ObjectiveFunctions::getObjectiveFunction(objFuncType_);
        
        // 创建并配置优化器
        ICOOTOptimizer optimizer;
        optimizer.setParameters(populationSize_, maxIterations_);
        optimizer.setShowProgress(showProgress_);
        
        // 执行优化
        auto [bestSolution, bestFitness, convergenceCurve] = optimizer.optimize(
            objFunc, dim, lb, ub, grayImage, numThresholds_
        );
        
        // 排序阈值
        std::sort(bestSolution.begin(), bestSolution.end());
        
        // 设置输出
        outThresholds = bestSolution;
        outFitness = bestFitness;
        outConvergenceCurve = convergenceCurve;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in ICOOT segmentation: " << e.what() << std::endl;
        return false;
    }
}

} // namespace ICOOT
