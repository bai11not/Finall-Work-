#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>

namespace ICOOT {

// ICOOT优化算法类
class ICOOTOptimizer {
public:
    // 构造函数
    ICOOTOptimizer();
    
    // 设置算法参数
    void setParameters(int populationSize = 50, int maxIterations = 100,
                       double alpha = 0.1, double beta = 0.05, double aInit = 2.0);
    
    // 优化函数
    // objFunc: 目标函数
    // dim: 优化维度（阈值数量）
    // lb: 下界
    // ub: 上界
    // image: 输入图像
    // numThresholds: 阈值数量
    // 返回值: 最优解、最优适应度值、收敛曲线
    std::tuple<std::vector<double>, double, std::vector<double>> optimize(
        const std::function<double(const std::vector<double>&, const cv::Mat&, int)>& objFunc,
        int dim,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        const cv::Mat& image,
        int numThresholds
    );
    
    // 获取最佳适应度值
    double getBestFitness() const;
    
    // 获取收敛曲线
    std::vector<double> getConvergenceCurve() const;
    
    // 设置是否显示迭代信息
    void setShowProgress(bool showProgress);

private:
    // 算法参数
    int populationSize_;  // 种群大小
    int maxIterations_;   // 最大迭代次数
    double alpha_;        // 局部搜索率
    double beta_;         // 随机性参数
    double aInit_;        // 初始a参数
    bool showProgress_;   // 是否显示迭代信息
    
    // 结果
    double bestFitness_;               // 最佳适应度值
    std::vector<double> convergenceCurve_;  // 收敛曲线
    
    // 初始化种群
    std::vector<std::vector<double>> initializePopulation(int populationSize, int dim,
                                                         const std::vector<double>& lb,
                                                         const std::vector<double>& ub);
    
    // 计算适应度
    std::vector<double> evaluatePopulation(const std::vector<std::vector<double>>& population,
                                          const std::function<double(const std::vector<double>&, const cv::Mat&, int)>& objFunc,
                                          const cv::Mat& image, int numThresholds);
    
    // 边界处理
    void enforceBounds(std::vector<double>& individual, const std::vector<double>& lb,
                      const std::vector<double>& ub);
    
    // 精英保留
    void elitism(std::vector<std::vector<double>>& population,
                std::vector<double>& fitness, int eliteCount);
};

} // namespace ICOOT
