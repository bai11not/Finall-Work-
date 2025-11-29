#include "../include/ICOOTOptimizer.h"
#include "../include/ImageProcessor.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace ICOOT {

ICOOTOptimizer::ICOOTOptimizer() {
    // 设置默认参数
    setParameters();
}

void ICOOTOptimizer::setParameters(int populationSize, int maxIterations,
                                  double alpha, double beta, double aInit) {
    populationSize_ = populationSize;
    maxIterations_ = maxIterations;
    alpha_ = alpha;
    beta_ = beta;
    aInit_ = aInit;
    showProgress_ = true;
    bestFitness_ = 0.0;
    convergenceCurve_.clear();
}

std::tuple<std::vector<double>, double, std::vector<double>> ICOOTOptimizer::optimize(
    const std::function<double(const std::vector<double>&, const cv::Mat&, int)>& objFunc,
    int dim,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    const cv::Mat& image,
    int numThresholds
) {
    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> normalDis(0.0, 1.0);
    
    // 初始化种群
    std::vector<std::vector<double>> population = initializePopulation(populationSize_, dim, lb, ub);
    
    // 计算初始适应度
    std::vector<double> fitness = evaluatePopulation(population, objFunc, image, numThresholds);
    
    // 找到初始最优解
    auto bestIdx = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
    bestFitness_ = fitness[bestIdx];
    std::vector<double> bestSolution = population[bestIdx];
    
    // 初始化收敛曲线
    convergenceCurve_.resize(maxIterations_, 0.0);
    
    // 主迭代循环
    for (int iter = 0; iter < maxIterations_; ++iter) {
        // 更新a参数（自适应调整）
        double a = aInit_ - iter * (aInit_ / maxIterations_);
        
        // 对每个个体进行更新
        for (int i = 0; i < populationSize_; ++i) {
            // 随机选择两种行为：探索或开发
            double r = dis(gen);
            std::vector<double> newPosition = population[i];
            
            if (r < 0.5) { // 探索行为
                double r1 = dis(gen);
                double r2 = dis(gen);
                
                if (r1 < 0.5) {
                    // 基于种群均值的探索
                    std::vector<double> meanPop(dim, 0.0);
                    for (const auto& individual : population) {
                        for (int j = 0; j < dim; ++j) {
                            meanPop[j] += individual[j];
                        }
                    }
                    for (int j = 0; j < dim; ++j) {
                        meanPop[j] /= populationSize_;
                    }
                    
                    for (int j = 0; j < dim; ++j) {
                        newPosition[j] += a * (r2 - 0.5) * (meanPop[j] - newPosition[j]);
                    }
                } else {
                    // 基于全局最优的探索
                    for (int j = 0; j < dim; ++j) {
                        newPosition[j] += a * (r2 - 0.5) * (bestSolution[j] - newPosition[j]);
                    }
                }
            } else { // 开发行为
                double r1 = dis(gen);
                double r2 = dis(gen);
                
                // 自适应局部搜索
                if (r1 < alpha_) {
                    // 局部搜索
                    for (int j = 0; j < dim; ++j) {
                        newPosition[j] += beta_ * normalDis(gen);
                    }
                } else {
                    // 基于当前最优的开发
                    for (int j = 0; j < dim; ++j) {
                        newPosition[j] += r2 * (bestSolution[j] - newPosition[j]);
                    }
                }
            }
            
            // 边界处理
            enforceBounds(newPosition, lb, ub);
            
            // 确保阈值有序
            std::sort(newPosition.begin(), newPosition.end());
            
            // 计算新的适应度
            double newFitness = objFunc(newPosition, image, numThresholds);
            
            // 贪婪选择
            if (newFitness < fitness[i]) {
                fitness[i] = newFitness;
                population[i] = newPosition;
                
                // 更新全局最优
                if (newFitness < bestFitness_) {
                    bestFitness_ = newFitness;
                    bestSolution = newPosition;
                }
            }
        }
        
        // 精英保留策略
        int eliteCount = std::max(1, static_cast<int>(0.1 * populationSize_));
        elitism(population, fitness, eliteCount);
        
        // 记录收敛曲线
        convergenceCurve_[iter] = bestFitness_;
        
        // 显示迭代信息
        if (showProgress_) {
            std::cout << "Iteration " << (iter + 1) << "/" << maxIterations_ 
                      << ": Best Fitness = " << bestFitness_ << std::endl;
        }
    }
    
    return {bestSolution, bestFitness_, convergenceCurve_};
}

double ICOOTOptimizer::getBestFitness() const {
    return bestFitness_;
}

std::vector<double> ICOOTOptimizer::getConvergenceCurve() const {
    return convergenceCurve_;
}

void ICOOTOptimizer::setShowProgress(bool showProgress) {
    showProgress_ = showProgress;
}

std::vector<std::vector<double>> ICOOTOptimizer::initializePopulation(int populationSize, int dim,
                                                                   const std::vector<double>& lb,
                                                                   const std::vector<double>& ub) {
    std::vector<std::vector<double>> population(populationSize, std::vector<double>(dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::uniform_real_distribution<> dis(lb[j], ub[j]);
            population[i][j] = dis(gen);
        }
        // 确保阈值有序
        std::sort(population[i].begin(), population[i].end());
    }
    
    return population;
}

std::vector<double> ICOOTOptimizer::evaluatePopulation(const std::vector<std::vector<double>>& population,
                                                     const std::function<double(const std::vector<double>&, const cv::Mat&, int)>& objFunc,
                                                     const cv::Mat& image, int numThresholds) {
    std::vector<double> fitness(population.size());
    
    for (int i = 0; i < population.size(); ++i) {
        fitness[i] = objFunc(population[i], image, numThresholds);
    }
    
    return fitness;
}

void ICOOTOptimizer::enforceBounds(std::vector<double>& individual, const std::vector<double>& lb,
                                 const std::vector<double>& ub) {
    for (int i = 0; i < individual.size(); ++i) {
        individual[i] = std::max(lb[i], std::min(individual[i], ub[i]));
    }
}

void ICOOTOptimizer::elitism(std::vector<std::vector<double>>& population,
                           std::vector<double>& fitness, int eliteCount) {
    // 创建索引数组
    std::vector<int> indices(fitness.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 根据适应度排序索引
    std::sort(indices.begin(), indices.end(), [&fitness](int a, int b) {
        return fitness[a] < fitness[b];
    });
    
    // 保存精英个体
    std::vector<std::vector<double>> elitePop(eliteCount);
    std::vector<double> eliteFitness(eliteCount);
    
    for (int i = 0; i < eliteCount; ++i) {
        elitePop[i] = population[indices[i]];
        eliteFitness[i] = fitness[indices[i]];
    }
    
    // 将精英放回种群的前eliteCount个位置
    for (int i = 0; i < eliteCount; ++i) {
        population[i] = elitePop[i];
        fitness[i] = eliteFitness[i];
    }
}

} // namespace ICOOT
