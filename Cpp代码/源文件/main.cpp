#include "D:/jisuanfangfa/cpp/include/ImageProcessor.h"
#include "D:/jisuanfangfa/cpp/include/ICOOTOptimizer.h"
#include "D:/jisuanfangfa/cpp/include/BatchProcessor.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace fs = std::filesystem;

// 显示帮助信息
void showHelp() {
    std::cout << "\nICOOT Segmentation Program - C++ Implementation" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Usage: " << std::endl;
    std::cout << "  Single image processing: " << std::endl;
    std::cout << "    icoot_segmentation --image <image_path> --output <output_dir> [options]" << std::endl;
    std::cout << "  Dataset processing: " << std::endl;
    std::cout << "    icoot_segmentation --dataset <dataset_path> --output <output_dir> [options]" << std::endl;
    std::cout << "\nOptions: " << std::endl;
    std::cout << "  -t, --thresholds <n>     Number of thresholds (default: 3)" << std::endl;
    std::cout << "  -f, --function <type>    Objective function type: entropy, otsu, cross (default: entropy)" << std::endl;
    std::cout << "  -p, --population <n>     Population size for ICOOT (default: 30)" << std::endl;
    std::cout << "  -i, --iterations <n>     Maximum number of iterations (default: 100)" << std::endl;
    std::cout << "  -e, --edge-style         Output edge detection style segmentation (default: region style)" << std::endl;
    std::cout << "  -q, --quiet              Disable progress display" << std::endl;
    std::cout << "  -h, --help               Show this help message" << std::endl;
    std::cout << "\nExamples: " << std::endl;
    std::cout << "  icoot_segmentation --image sample.jpg --output results --thresholds 4 --function otsu" << std::endl;
    std::cout << "  icoot_segmentation --dataset images_folder --output results_dir --iterations 200 --edge-style" << std::endl;
    std::cout << std::endl;
}

// 解析命令行参数
bool parseArguments(int argc, char* argv[], std::string& inputPath, std::string& outputPath,
                   bool& isDataset, int& numThresholds, ICOOT::ObjectiveFunctionType& objFuncType,
                   int& populationSize, int& maxIterations, bool& outputEdgeStyle, bool& showProgress) {
    // 设置默认值
    numThresholds = 300;
    objFuncType = ICOOT::ObjectiveFunctionType::MAX_ENTROPY;
    populationSize = 30;
    maxIterations = 100;
    outputEdgeStyle = false;
    showProgress = true;
    isDataset = false;
    
    // 处理参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            showHelp();
            return false;
        }
        else if (arg == "--image") {
            if (i + 1 < argc) {
                inputPath = argv[++i];
                isDataset = false;
            } else {
                std::cerr << "Error: --image requires a path argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--dataset") {
            if (i + 1 < argc) {
                inputPath = argv[++i];
                isDataset = true;
            } else {
                std::cerr << "Error: --dataset requires a path argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                outputPath = argv[++i];
            } else {
                std::cerr << "Error: --output requires a path argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--thresholds" || arg == "-t") {
            if (i + 1 < argc) {
                numThresholds = std::stoi(argv[++i]);
                if (numThresholds < 1) {
                    std::cerr << "Error: Number of thresholds must be at least 1." << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: --thresholds requires a number argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--function" || arg == "-f") {
            if (i + 1 < argc) {
                std::string funcType = argv[++i];
                std::transform(funcType.begin(), funcType.end(), funcType.begin(), ::tolower);
                
                if (funcType == "entropy") {
                    objFuncType = ICOOT::ObjectiveFunctionType::MAX_ENTROPY;
                } else if (funcType == "otsu") {
                    objFuncType = ICOOT::ObjectiveFunctionType::OTSU;
                } else if (funcType == "cross") {
                    objFuncType = ICOOT::ObjectiveFunctionType::CROSS_ENTROPY;
                } else {
                    std::cerr << "Error: Invalid objective function type. Use 'entropy', 'otsu', or 'cross'." << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: --function requires a type argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--population" || arg == "-p") {
            if (i + 1 < argc) {
                populationSize = std::stoi(argv[++i]);
                if (populationSize < 5) {
                    std::cerr << "Error: Population size must be at least 5." << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: --population requires a number argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--iterations" || arg == "-i") {
            if (i + 1 < argc) {
                maxIterations = std::stoi(argv[++i]);
                if (maxIterations < 1) {
                    std::cerr << "Error: Number of iterations must be at least 1." << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: --iterations requires a number argument." << std::endl;
                return false;
            }
        }
        else if (arg == "--edge-style" || arg == "-e") {
            outputEdgeStyle = true;
        }
        else if (arg == "--quiet" || arg == "-q") {
            showProgress = false;
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            showHelp();
            return false;
        }
    }
    
    // 验证必要参数
    if (inputPath.empty()) {
        std::cerr << "Error: No input path specified. Use --image or --dataset." << std::endl;
        showHelp();
        return false;
    }
    
    if (outputPath.empty()) {
        std::cerr << "Error: No output path specified. Use --output." << std::endl;
        showHelp();
        return false;
    }
    
    // 验证输入路径
    if (isDataset) {
        if (!fs::exists(inputPath) || !fs::is_directory(inputPath)) {
            std::cerr << "Error: Dataset directory does not exist or is not a directory: " << inputPath << std::endl;
            return false;
        }
    } else {
        if (!fs::exists(inputPath) || fs::is_directory(inputPath)) {
            std::cerr << "Error: Image file does not exist or is a directory: " << inputPath << std::endl;
            return false;
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    // 设置控制台编码为UTF-8
    SetConsoleOutputCP(CP_UTF8);
    
    std::cout << "Starting ICOOT Segmentation Program..." << std::endl;
    
    // 检查参数数量
    if (argc < 2) {
        showHelp();
        return 1;
    }
    
    // 解析参数
    std::string inputPath, outputPath;
    bool isDataset;
    int numThresholds, populationSize, maxIterations;
    ICOOT::ObjectiveFunctionType objFuncType;
    bool outputEdgeStyle, showProgress;
    
    if (!parseArguments(argc, argv, inputPath, outputPath, isDataset, numThresholds, objFuncType,
                       populationSize, maxIterations, outputEdgeStyle, showProgress)) {
        return 1;
    }
    
    // 创建输出目录（如果不存在）
    try {
        if (!fs::exists(outputPath)) {
            std::cout << "Creating output directory: " << outputPath << std::endl;
            if (!fs::create_directories(outputPath)) {
                std::cerr << "Error: Failed to create output directory: " << outputPath << std::endl;
                std::cerr << "Attempting to use current directory as fallback..." << std::endl;
                outputPath = ".";
            }
        }
        
        // 验证输出目录的写入权限
        std::string testFilePath = outputPath + "\\" + "test_write_permission.txt";
        std::ofstream testFile(testFilePath);
        bool hasWritePermission = testFile.good();
        testFile.close();
        if (hasWritePermission) {
            fs::remove(testFilePath);
            std::cout << "[SUCCESS] Output directory has write permission: " << outputPath << std::endl;
        } else {
            std::cerr << "[WARNING] No write permission to output directory: " << outputPath << std::endl;
            std::cerr << "Attempting to use current directory as fallback..." << std::endl;
            outputPath = ".";
        }
        
        std::cout << "Final output directory: " << outputPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return 1;
    }
    
    // 初始化批处理器
    ICOOT::BatchProcessor processor;
    processor.setParameters(numThresholds, objFuncType, populationSize, maxIterations, 
                           outputEdgeStyle, showProgress);
    // 设置保存中间数据为true，确保图像被保存
    processor.setSaveIntermediateData(true);
    
    std::cout << "\nStarting image processing..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    bool success = false;
    try {
        if (isDataset) {
            // 处理数据集
            std::cout << "Processing dataset: " << inputPath << std::endl;
            success = processor.processDataset(inputPath, outputPath);
        } else {
            // 处理单个图像
            std::cout << "Processing single image: " << inputPath << std::endl;
            success = processor.processSingleImage(inputPath, outputPath);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "----------------------------------------" << std::endl;
    
    if (!success) {
        std::cerr << "Error: Processing failed" << std::endl;
        
        // 输出可能的解决方法
        std::cout << "\nPossible solutions:" << std::endl;
        std::cout << "1. Check if input image exists and is accessible" << std::endl;
        std::cout << "2. Ensure OpenCV is properly installed and linked" << std::endl;
        std::cout << "3. Verify you have write permissions to output directory" << std::endl;
        std::cout << "4. Try using a different output directory" << std::endl;
        
        return 1;
    } else {
        std::cout << "\n✅ Processing completed successfully!" << std::endl;
        std::cout << "Check the output directory for segmented images: " << outputPath << std::endl;
        
        // 列出输出目录中的PNG文件
        std::cout << "\nSegmented images found:" << std::endl;
        try {
            int imageCount = 0;
            for (const auto& entry : fs::directory_iterator(outputPath)) {
                if (entry.is_regular_file() && entry.path().extension() == ".png") {
                    std::cout << "  - " << entry.path().filename() << std::endl;
                    imageCount++;
                }
            }
            if (imageCount == 0) {
                std::cout << "  No PNG images found in output directory." << std::endl;
                std::cout << "  Check if images were saved in alternative formats (.bmp) or current directory." << std::endl;
            } else {
                std::cout << "Found " << imageCount << " segmented images." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error listing output files: " << e.what() << std::endl;
        }
        return 0;
    }
}
