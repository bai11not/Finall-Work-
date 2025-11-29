function [bestSolution, bestFitness, convergenceCurve] = ICOOT(objFunc, dim, nPop, maxIter, lb, ub, image, numThresholds)
% ICOOT - 改进的COOT优化算法 (Improved COOT Optimization)
% 用于多级阈值分割的优化
% 
% 输入参数:
%   objFunc - 目标函数句柄
%   dim - 优化问题维度 (阈值数量)
%   nPop - 种群大小
%   maxIter - 最大迭代次数
%   lb - 变量下界
%   ub - 变量上界
%   image - 输入图像
%   numThresholds - 阈值数量
% 
% 输出参数:
%   bestSolution - 最优解 (阈值向量)
%   bestFitness - 最优适应度值
%   convergenceCurve - 收敛曲线


    % 初始化种群
    pop = zeros(nPop, dim);
    for i = 1:nPop
        % 在搜索空间中随机初始化每个个体
        pop(i, :) = lb + rand(1, dim) .* (ub - lb);
    end
    
    % 计算初始适应度
    fitness = zeros(nPop, 1);
    for i = 1:nPop
        % 确保阈值有序
        sortedPop = sort(pop(i, :));
        fitness(i) = objFunc(sortedPop, image, numThresholds);
        pop(i, :) = sortedPop; % 更新为有序的阈值
    end
    
    % 找到初始最优解
    [bestFitness, bestIdx] = min(fitness); % 假设目标是最小化
    bestSolution = pop(bestIdx, :);
    convergenceCurve = zeros(maxIter, 1);
    
    % 初始化参数
    a = 2; % 控制探索范围的参数
    alpha = 0.1; % 局部搜索率
    beta = 0.05; % 随机性参数
    
    % 主迭代循环
    for iter = 1:maxIter
        % 更新a参数（自适应调整）
        a = 2 - iter * (2 / maxIter);
        
        % 对每个个体进行更新
        for i = 1:nPop
            % 随机选择两种行为：探索或开发
            r = rand();
            
            if r < 0.5 % 探索行为
                % 改进的探索策略 - 引入更多多样性
                r1 = rand();
                r2 = rand();
                
                if r1 < 0.5
                    % 基于种群均值的探索
                    meanPop = mean(pop);
                    pop(i, :) = pop(i, :) + a .* (r2 - 0.5) .* (meanPop - pop(i, :));
                else
                    % 基于全局最优的探索
                    pop(i, :) = pop(i, :) + a .* (r2 - 0.5) .* (bestSolution - pop(i, :));
                end
            else % 开发行为
                % 改进的开发策略 - 增强局部搜索能力
                r1 = rand();
                r2 = rand();
                
                % 自适应局部搜索
                if r1 < alpha
                    % 局部搜索
                    pop(i, :) = pop(i, :) + beta .* randn(1, dim);
                else
                    % 基于当前最优的开发
                    pop(i, :) = pop(i, :) + r2 .* (bestSolution - pop(i, :));
                end
            end
            
            % 边界处理
            pop(i, :) = max(pop(i, :), lb);
            pop(i, :) = min(pop(i, :), ub);
            
            % 确保阈值有序
            sortedPop = sort(pop(i, :));
            
            % 计算新的适应度
            newFitness = objFunc(sortedPop, image, numThresholds);
            
            % 贪婪选择
            if newFitness < fitness(i)
                fitness(i) = newFitness;
                pop(i, :) = sortedPop;
                
                % 更新全局最优
                if newFitness < bestFitness
                    bestFitness = newFitness;
                    bestSolution = sortedPop;
                end
            end
        end
        
        % 精英保留策略（改进点）
        [sortedFitness, sortedIdx] = sort(fitness);
        eliteCount = max(1, round(0.1 * nPop)); % 保留10%的精英
        
        % 确保精英个体保留
        pop(1:eliteCount, :) = pop(sortedIdx(1:eliteCount), :);
        fitness(1:eliteCount) = sortedFitness(1:eliteCount);
        
        % 记录收敛曲线
        convergenceCurve(iter) = bestFitness;
        
        % 显示迭代信息
        fprintf('迭代 %d/%d: 最佳适应度 = %.6f\n', iter, maxIter, bestFitness);
    end
end
