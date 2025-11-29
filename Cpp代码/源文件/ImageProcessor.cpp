#include "../include/ImageProcessor.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace ICOOT {

ImageProcessor::ImageProcessor() {
    // 设置控制台编码为UTF-8，确保中文正常显示
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
}

ImageProcessor::~ImageProcessor() {
}

bool ImageProcessor::loadImage(const std::string& imagePath) {
    originalImage_ = cv::imread(imagePath);
    if (originalImage_.empty()) {
        std::cerr << "Error: Cannot load image from " << imagePath << std::endl;
        return false;
    }
    convertToGrayscale();
    return true;
}

void ImageProcessor::convertToGrayscale() {
    if (originalImage_.channels() > 1) {
        cv::cvtColor(originalImage_, grayscaleImage_, cv::COLOR_BGR2GRAY);
    } else {
        grayscaleImage_ = originalImage_.clone();
    }
    // 归一化到[0,1]范围
    grayscaleImage_.convertTo(grayscaleImage_, CV_64F, 1.0 / 255.0);
}

cv::Mat ImageProcessor::getGrayscaleImage() const {
    return grayscaleImage_;
}

cv::Mat ImageProcessor::getOriginalImage() const {
    std::cout << "[ImageProcessor::getOriginalImage] 返回原始图像，尺寸: " << originalImage_.rows << "x" << originalImage_.cols << std::endl;
    return originalImage_;
}

cv::Mat ImageProcessor::applyThresholds(const std::vector<double>& thresholds) {
    std::vector<double> sortedThresholds = thresholds;
    std::sort(sortedThresholds.begin(), sortedThresholds.end());
    
    cv::Mat segmentedImage = cv::Mat::zeros(grayscaleImage_.size(), CV_64F);
    int numRegions = static_cast<int>(sortedThresholds.size()) + 1;
    
    for (int i = 0; i < sortedThresholds.size(); ++i) {
        if (i == 0) {
            // 第一个区域
            segmentedImage.setTo(static_cast<double>(i + 1) / numRegions, grayscaleImage_ <= sortedThresholds[i]);
        } else {
            // 中间区域
            segmentedImage.setTo(static_cast<double>(i + 1) / numRegions, 
                                (grayscaleImage_ > sortedThresholds[i-1]) & (grayscaleImage_ <= sortedThresholds[i]));
        }
    }
    
    // 最后一个区域
    segmentedImage.setTo(1.0, grayscaleImage_ > sortedThresholds.back());
    
    return segmentedImage;
}

cv::Mat ImageProcessor::generateEdgeStyleSegmentation(const std::vector<double>& thresholds) {
    cv::Mat segmentedImage = applyThresholds(thresholds);
    int rows = segmentedImage.rows;
    int cols = segmentedImage.cols;
    
    // 1. 轻微高斯模糊
    cv::Mat blurredImage = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            blurredImage.at<double>(i, j) = (
                segmentedImage.at<double>(i-1, j-1) + segmentedImage.at<double>(i-1, j) + segmentedImage.at<double>(i-1, j+1) +
                segmentedImage.at<double>(i, j-1) + 2 * segmentedImage.at<double>(i, j) + segmentedImage.at<double>(i, j+1) +
                segmentedImage.at<double>(i+1, j-1) + segmentedImage.at<double>(i+1, j) + segmentedImage.at<double>(i+1, j+1)
            ) / 10.0;
        }
    }
    
    // 2. Sobel边缘检测
    cv::Mat gradientMag = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            double Gx = 2 * blurredImage.at<double>(i-1, j+1) + 4 * blurredImage.at<double>(i, j+1) + 2 * blurredImage.at<double>(i+1, j+1) -
                        2 * blurredImage.at<double>(i-1, j-1) - 4 * blurredImage.at<double>(i, j-1) - 2 * blurredImage.at<double>(i+1, j-1);
            double Gy = 2 * blurredImage.at<double>(i+1, j-1) + 4 * blurredImage.at<double>(i+1, j) + 2 * blurredImage.at<double>(i+1, j+1) -
                        2 * blurredImage.at<double>(i-1, j-1) - 4 * blurredImage.at<double>(i-1, j) - 2 * blurredImage.at<double>(i-1, j+1);
            
            gradientMag.at<double>(i, j) = std::sqrt(Gx * Gx + Gy * Gy);
        }
    }
    
    // 3. 计算边缘阈值
    double maxGradient = 0.0;
    double meanGradient = 0.0;
    double stdGradient = 0.0;
    
    // 计算最大梯度值
    cv::minMaxLoc(gradientMag, nullptr, &maxGradient);
    
    // 计算平均梯度和标准差
    meanGradient = cv::mean(gradientMag)[0];
    
    cv::Mat diffSquared = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double diff = gradientMag.at<double>(i, j) - meanGradient;
            diffSquared.at<double>(i, j) = diff * diff;
        }
    }
    stdGradient = std::sqrt(cv::mean(diffSquared)[0]);
    
    // 设置阈值
    double threshold = 0.15 * maxGradient + 0.1 * meanGradient - 0.05 * stdGradient;
    threshold = (threshold > 0.02 * maxGradient) ? threshold : 0.02 * maxGradient;
    
    // 应用阈值
    cv::Mat edgeImage = cv::Mat::zeros(rows, cols, CV_64F);
    edgeImage.setTo(1.0, gradientMag > threshold);
    
    // 4. 边缘细化
    cv::Mat thinEdgeImage = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (edgeImage.at<double>(i, j) == 1.0) {
                // 检查邻域
                bool hasNeighbor = false;
                for (int di = -1; di <= 1 && !hasNeighbor; ++di) {
                    for (int dj = -1; dj <= 1 && !hasNeighbor; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        if (edgeImage.at<double>(i + di, j + dj) == 1.0) {
                            hasNeighbor = true;
                        }
                    }
                }
                if (hasNeighbor) {
                    thinEdgeImage.at<double>(i, j) = 1.0;
                }
            }
        }
    }
    edgeImage = thinEdgeImage;
    
    // 反转边缘图像，白底黑线
    edgeImage = 1.0 - edgeImage;
    
    // 5. 对比度增强
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            edgeImage.at<double>(i, j) = std::pow(edgeImage.at<double>(i, j), 0.4);
        }
    }
    
    // 6. 后处理：强化边缘
    edgeImage = edgeImage * 1.1;
    edgeImage.setTo(0.0, edgeImage < 0.2);
    
    // 确保值在[0,1]范围内
    cv::threshold(edgeImage, edgeImage, 1.0, 1.0, cv::THRESH_TRUNC);
    edgeImage.setTo(0.0, edgeImage < 0.0);
    
    return edgeImage;
}

std::vector<double> ImageProcessor::calculateHistogram(int bins) {
    std::vector<double> histogram(bins, 0.0);
    double binWidth = 1.0 / bins;
    
    for (int i = 0; i < grayscaleImage_.rows; ++i) {
        for (int j = 0; j < grayscaleImage_.cols; ++j) {
            double pixelValue = grayscaleImage_.at<double>(i, j);
            int binIndex = static_cast<int>(pixelValue / binWidth);
            // 处理边界情况
            if (binIndex >= bins) binIndex = bins - 1;
            if (binIndex < 0) binIndex = 0;
            histogram[binIndex]++;
        }
    }
    
    // 归一化
    double totalPixels = grayscaleImage_.rows * grayscaleImage_.cols;
    for (double& value : histogram) {
        value /= totalPixels;
    }
    
    return histogram;
}

bool ImageProcessor::saveImage(const cv::Mat& image, const std::string& filePath) {
    std::cout << "[ImageProcessor::saveImage] Attempting to save image to: " << filePath << std::endl;
    
    // 检查输入图像是否有效
    if (image.empty()) {
        std::cerr << "[ERROR] Image is empty, cannot save!" << std::endl;
        return false;
    }
    
    std::cout << "[ImageProcessor::saveImage] Input image: rows=" << image.rows 
              << ", cols=" << image.cols 
              << ", channels=" << image.channels() 
              << ", type=" << image.type() << std::endl;

    // 将图像转换为8位无符号整型（适合保存为图像文件）
    cv::Mat saveImage;
    if (image.type() == CV_64F) {
        std::cout << "[ImageProcessor::saveImage] Converting CV_64F to CV_8U" << std::endl;
        // 先尝试找到非零的最大值和最小值进行归一化
        double minVal, maxVal;
        cv::minMaxLoc(image, &minVal, &maxVal);
        
        // 避免除以零的情况
        if (maxVal > minVal) {
            saveImage = (image - minVal) / (maxVal - minVal); // 归一化到[0,1]
        } else {
            saveImage = image.clone();
        }
        
        saveImage.convertTo(saveImage, CV_8U, 255.0);
    } else if (image.type() == CV_32F) {
        std::cout << "[ImageProcessor::saveImage] Converting CV_32F to CV_8U" << std::endl;
        // 归一化到[0, 1]范围
        cv::normalize(image, saveImage, 0, 1, cv::NORM_MINMAX);
        // 转换为8位无符号整型
        saveImage.convertTo(saveImage, CV_8U, 255.0);
    } else if (image.type() == CV_32S || image.type() == CV_16S || image.type() == CV_16U) {
        std::cout << "[ImageProcessor::saveImage] Converting integer type to CV_8U" << std::endl;
        // 归一化并转换为8位
        cv::normalize(image, saveImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    } else {
        std::cout << "[ImageProcessor::saveImage] Using image as-is (type = " << image.type() << ")" << std::endl;
        saveImage = image.clone();
    }
    
    // 确保图像至少是单通道的
    if (saveImage.channels() == 0) {
        std::cerr << "[ERROR] Invalid image channels count!" << std::endl;
        return false;
    }
    
    std::cout << "[ImageProcessor::saveImage] Save image prepared: rows=" << saveImage.rows 
              << ", cols=" << saveImage.cols 
              << ", channels=" << saveImage.channels() 
              << ", type=" << saveImage.type() << std::endl;

    // 尝试保存图像
    bool success = cv::imwrite(filePath, saveImage);
    
    if (!success) {
        std::cerr << "[ERROR] Failed to save image to " << filePath << std::endl;
        
        // 输出详细的错误信息
        std::cerr << "Possible reasons:" << std::endl;
        std::cerr << "1. Output directory does not exist" << std::endl;
        std::cerr << "2. No write permissions" << std::endl;
        std::cerr << "3. OpenCV might not be built with PNG support" << std::endl;
        std::cerr << "4. Invalid image format" << std::endl;
        
        // 尝试保存到当前目录作为备选
        size_t lastSlash = filePath.find_last_of("\\/");
        std::string fileName = (lastSlash != std::string::npos) ? filePath.substr(lastSlash + 1) : filePath;
        std::string altPath = fileName;
        
        std::cout << "[ImageProcessor::saveImage] Attempting alternative save path: " << altPath << std::endl;
        
        // 尝试不同的图像格式
        std::string altPathBmp = altPath.substr(0, altPath.find_last_of(".")) + ".bmp";
        success = cv::imwrite(altPathBmp, saveImage);
        
        if (success) {
            std::cout << "[SUCCESS] Image saved successfully to alternative path: " << altPathBmp << std::endl;
            return true;
        } else {
            std::cerr << "[ERROR] Failed to save to alternative path as well!" << std::endl;
            return false;
        }
    }

    std::cout << "[SUCCESS] Image saved successfully to: " << filePath << std::endl;
    return true;
}

// ObjectiveFunctions 类实现
std::pair<std::vector<double>, std::vector<double>> ObjectiveFunctions::calculateHistogramAndCenters(const cv::Mat& grayImage, int bins) {
    std::vector<double> histogram(bins, 0.0);
    std::vector<double> binCenters(bins);
    double binWidth = 1.0 / bins;
    
    // 计算直方图
    for (int i = 0; i < grayImage.rows; ++i) {
        for (int j = 0; j < grayImage.cols; ++j) {
            double pixelValue = grayImage.at<double>(i, j);
            int binIndex = static_cast<int>(pixelValue / binWidth);
            if (binIndex >= bins) binIndex = bins - 1;
            if (binIndex < 0) binIndex = 0;
            histogram[binIndex]++;
        }
    }
    
    // 归一化直方图
    double totalPixels = grayImage.rows * grayImage.cols;
    for (double& value : histogram) {
        value /= totalPixels;
    }
    
    // 计算bin中心
    for (int i = 0; i < bins; ++i) {
        binCenters[i] = (i + 0.5) * binWidth;
    }
    
    return {histogram, binCenters};
}

double ObjectiveFunctions::maxEntropyObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds) {
    std::vector<double> sortedThresholds = thresholds;
    std::sort(sortedThresholds.begin(), sortedThresholds.end());
    
    auto [histogram, binCenters] = calculateHistogramAndCenters(grayImage, 256);
    
    double totalEntropy = 0.0;
    double prevThresh = 0.0;
    
    for (int i = 0; i < numThresholds; ++i) {
        double currentThresh = sortedThresholds[i];
        double sumProb = 0.0;
        double entropy = 0.0;
        
        for (int j = 0; j < binCenters.size(); ++j) {
            if (binCenters[j] > prevThresh && binCenters[j] <= currentThresh) {
                sumProb += histogram[j];
            }
        }
        
        if (sumProb > 0.0) {
            for (int j = 0; j < binCenters.size(); ++j) {
                if (binCenters[j] > prevThresh && binCenters[j] <= currentThresh) {
                    double prob = histogram[j] / sumProb;
                    if (prob > 0.0) {
                        entropy -= prob * std::log2(prob);
                    }
                }
            }
            totalEntropy += entropy;
        }
        
        prevThresh = currentThresh;
    }
    
    // 处理最后一个区域
    double sumProb = 0.0;
    double entropy = 0.0;
    for (int j = 0; j < binCenters.size(); ++j) {
        if (binCenters[j] > prevThresh) {
            sumProb += histogram[j];
        }
    }
    
    if (sumProb > 0.0) {
        for (int j = 0; j < binCenters.size(); ++j) {
            if (binCenters[j] > prevThresh) {
                double prob = histogram[j] / sumProb;
                if (prob > 0.0) {
                    entropy -= prob * std::log2(prob);
                }
            }
        }
        totalEntropy += entropy;
    }
    
    // 返回负熵，因为ICOOT是最小化算法
    return -totalEntropy;
}

double ObjectiveFunctions::otsuObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds) {
    std::vector<double> sortedThresholds = thresholds;
    std::sort(sortedThresholds.begin(), sortedThresholds.end());
    
    auto [histogram, binCenters] = calculateHistogramAndCenters(grayImage, 256);
    
    int totalPixels = grayImage.rows * grayImage.cols;
    double globalMean = 0.0;
    
    // 计算全局均值
    for (int j = 0; j < binCenters.size(); ++j) {
        globalMean += binCenters[j] * histogram[j] * totalPixels;
    }
    globalMean /= totalPixels;
    
    double betweenClassVariance = 0.0;
    double prevThresh = -1.0;
    
    for (int i = 0; i < numThresholds; ++i) {
        double currentThresh = sortedThresholds[i];
        int lowerBin = (0 > static_cast<int>(prevThresh * 255) + 1) ? 0 : static_cast<int>(prevThresh * 255) + 1;
        int upperBin = (255 < static_cast<int>(currentThresh * 255) + 1) ? 255 : static_cast<int>(currentThresh * 255) + 1;
        
        if (upperBin >= lowerBin && lowerBin < 256 && upperBin < 256) {
            double currentWeight = 0.0;
            double currentMean = 0.0;
            
            for (int j = lowerBin; j <= upperBin; ++j) {
                currentWeight += histogram[j];
                currentMean += binCenters[j] * histogram[j];
            }
            
            if (currentWeight > 0.0) {
                currentMean /= currentWeight;
                betweenClassVariance += currentWeight * std::pow(currentMean - globalMean, 2);
            }
        }
        
        prevThresh = currentThresh;
    }
    
    // 处理最后一个类
    int lowerBin = (0 > static_cast<int>(prevThresh * 255) + 1) ? 0 : static_cast<int>(prevThresh * 255) + 1;
    if (lowerBin < 256) {
        double currentWeight = 0.0;
        double currentMean = 0.0;
        
        for (int j = lowerBin; j < 256; ++j) {
            currentWeight += histogram[j];
            currentMean += binCenters[j] * histogram[j];
        }
        
        if (currentWeight > 0.0) {
            currentMean /= currentWeight;
            betweenClassVariance += currentWeight * std::pow(currentMean - globalMean, 2);
        }
    }
    
    // 返回负的类间方差
    return -betweenClassVariance;
}

double ObjectiveFunctions::crossEntropyObjective(const std::vector<double>& thresholds, const cv::Mat& grayImage, int numThresholds) {
    std::vector<double> sortedThresholds = thresholds;
    std::sort(sortedThresholds.begin(), sortedThresholds.end());
    
    auto [histogram, binCenters] = calculateHistogramAndCenters(grayImage, 256);
    
    double crossEntropy = 0.0;
    double prevThresh = 0.0;
    
    for (int i = 0; i < numThresholds; ++i) {
        double currentThresh = sortedThresholds[i];
        double sumProb = 0.0;
        int count = 0;
        
        for (int j = 0; j < binCenters.size(); ++j) {
            if (binCenters[j] > prevThresh && binCenters[j] <= currentThresh) {
                sumProb += histogram[j];
                count++;
            }
        }
        
        if (count > 0 && sumProb > 0.0) {
            double idealProb = 1.0 / count;
            double entropy = 0.0;
            
            for (int j = 0; j < binCenters.size(); ++j) {
                if (binCenters[j] > prevThresh && binCenters[j] <= currentThresh) {
                    double prob = histogram[j] / sumProb;
                    if (prob > 0.0) {
                        entropy -= prob * std::log2(idealProb);
                    }
                }
            }
            crossEntropy += entropy;
        }
        
        prevThresh = currentThresh;
    }
    
    // 处理最后一个区域
    double sumProb = 0.0;
    int count = 0;
    for (int j = 0; j < binCenters.size(); ++j) {
        if (binCenters[j] > prevThresh) {
            sumProb += histogram[j];
            count++;
        }
    }
    
    if (count > 0 && sumProb > 0.0) {
        double idealProb = 1.0 / count;
        double entropy = 0.0;
        
        for (int j = 0; j < binCenters.size(); ++j) {
            if (binCenters[j] > prevThresh) {
                double prob = histogram[j] / sumProb;
                if (prob > 0.0) {
                    entropy -= prob * std::log2(idealProb);
                }
            }
        }
        crossEntropy += entropy;
    }
    
    return crossEntropy;
}

std::function<double(const std::vector<double>&, const cv::Mat&, int)> ObjectiveFunctions::getObjectiveFunction(ObjectiveFunctionType type) {
    switch (type) {
        case ObjectiveFunctionType::MAX_ENTROPY:
            return &maxEntropyObjective;
        case ObjectiveFunctionType::OTSU:
            return &otsuObjective;
        case ObjectiveFunctionType::CROSS_ENTROPY:
            return &crossEntropyObjective;
        default:
            return &maxEntropyObjective; // 默认返回最大熵
    }
}

} // namespace ICOOT
