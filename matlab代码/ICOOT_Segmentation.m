function [segmentedImage, thresholds, fitness, convergenceCurve] = ICOOT_Segmentation(image, numThresholds, objFuncType, varargin)
% ICOOT_Segmentation - 使用改进COOT优化算法进行多级阈值分割
% 
% 输入参数:
%   image - 输入图像（彩色或灰度）
%   numThresholds - 阈值数量
%   objFuncType - 目标函数类型: 'maxEntropy', 'otsu', 'crossEntropy'
%   varargin - 可选参数：
%     'PopulationSize' - 种群大小（默认50）
%     'MaxIterations' - 最大迭代次数（默认100）
%     'ShowProgress' - 是否显示进度（默认true）
%     'OutputEdgeStyle' - 是否输出边缘检测风格的结果（默认true）
% 
% 输出参数:
%   segmentedImage - 分割后的图像（边缘检测风格或多区域分割）
%   thresholds - 优化后的阈值
%   fitness - 最佳适应度值
%   convergenceCurve - 收敛曲线

    % 默认参数设置
    defaultParams = struct(...
        'PopulationSize', 50, ...
        'MaxIterations', 100, ...
        'ShowProgress', true, ...
        'OutputEdgeStyle', true);
    
    % 解析输入参数
    p = inputParser;
    addParameter(p, 'PopulationSize', defaultParams.PopulationSize, @isnumeric);
    addParameter(p, 'MaxIterations', defaultParams.MaxIterations, @isnumeric);
    addParameter(p, 'ShowProgress', defaultParams.ShowProgress, @islogical);
    addParameter(p, 'OutputEdgeStyle', defaultParams.OutputEdgeStyle, @islogical);
    parse(p, varargin{:});
    params = p.Results;
    
    % 保存原始彩色图像
    originalImage = image;
    
    % 转换为灰度用于阈值分割
    if size(image, 3) > 1
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end
    grayImage = im2double(grayImage);
    image = grayImage; % 用于分割算法的输入仍然是灰度图
    
    % 设置优化参数
    nPop = params.PopulationSize;
    maxIter = params.MaxIterations;
    dim = numThresholds;
    lb = zeros(1, dim) + 0.01; % 阈值下界，避免0
    ub = zeros(1, dim) + 0.99; % 阈值上界，避免1
    
    % 根据选择的目标函数类型创建目标函数句柄
    switch lower(objFuncType)
        case 'maxentropy'
            objFunc = @maxEntropyObjective;
        case 'otsu'
            objFunc = @otsuObjective;
        case 'crossentropy'
            objFunc = @crossEntropyObjective;
        otherwise
            error('不支持的目标函数类型。请选择: ''maxEntropy'', ''otsu'', 或 ''crossEntropy''');
    end
    
    % 调用ICOOT算法进行优化
    if params.ShowProgress
        disp(['使用ICOOT算法进行', num2str(numThresholds), '级阈值分割...']);
        disp(['目标函数: ', upper(objFuncType)]);
        disp(['图像尺寸: ', num2str(size(image))]);
        disp(['优化参数: 种群=', num2str(nPop), ', 迭代=', num2str(maxIter), ', 维度=', num2str(dim)]);
    end
    
    try
        [bestSolution, bestFitness, convergenceCurve] = ICOOT(...
            objFunc, dim, nPop, maxIter, lb, ub, grayImage, numThresholds);
    catch ME
        error('ICOOT算法执行错误: %s\n错误堆栈: %s', ME.message, getReport(ME));
    end
    
    % 排序并整理阈值
    thresholds = sort(bestSolution);
    fitness = bestFitness;
    
    % 应用阈值进行图像分割
    if params.OutputEdgeStyle
        % 生成边缘检测风格的结果
        segmentedImage = generateEdgeStyleSegmentation(grayImage, thresholds);
    else
        % 正常的多区域分割
        segmentedImage = applyThresholds(grayImage, thresholds);
    end
    
    if params.ShowProgress
        disp('阈值分割完成!');
        disp(['最佳阈值: ', num2str(thresholds)]);
        disp(['最佳适应度: ', num2str(fitness)]);
        
        % 在MATLAB中显示分割图像和原始图像进行对比
        figure('Name', 'ICOOT分割结果', 'Position', [100, 100, 800, 400]);
        
        % 显示原始图像（保留彩色）
        subplot(1, 2, 1);
        if size(originalImage, 3) > 1
            imshow(originalImage);
        else
            imshow(originalImage, []);
        end
        title('原始图像');
        
        % 显示分割图像
        subplot(1, 2, 2);
        if params.OutputEdgeStyle
            % 边缘检测风格显示（白底黑线）
            imshow(segmentedImage, []);
            colormap gray;
        else
            % 正常的多区域分割显示
            imshow(segmentedImage, []);
        end
        title(['分割结果 - ', upper(objFuncType)]);
        
        % 确保在分割图像的subplot中
        subplot(1, 2, 2);
        
        % 构建阈值文本并显示
        thresholdText = '阈值: ';
        for i = 1:length(thresholds)
            thresholdText = [thresholdText, sprintf('%.4f', thresholds(i))];
            if i < length(thresholds)
                thresholdText = [thresholdText, ', '];
            end
        end
        
        % 在分割图像上添加文本标注
        text(10, 25, thresholdText, 'Color', 'white', 'BackgroundColor', 'black', ...
             'FontSize', 10, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        
        % 显示适应度值
        fitnessText = sprintf('适应度: %.6f', fitness);
        text(10, 50, fitnessText, 'Color', 'white', 'BackgroundColor', 'black', ...
             'FontSize', 10, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    end
end

function fitness = maxEntropyObjective(thresholds, grayImage, numThresholds)
% 最大熵目标函数
    % 排序阈值
    thresholds = sort(thresholds);
    
    % 创建[0,1]范围内的直方图（使用256个bin）
    binEdges = linspace(0, 1, 257);
    histCounts = histcounts(grayImage(:), binEdges);
    histCounts = histCounts / sum(histCounts); % 归一化
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
    
    % 计算各个区域的熵
    totalEntropy = 0;
    prevThresh = 0;
    
    for i = 1:numThresholds
        % 找到当前阈值范围内的像素
        indices = binCenters > prevThresh & binCenters <= thresholds(i);
        if any(indices)
            prob = histCounts(indices) / sum(histCounts(indices));
            prob = prob(prob > 0); % 移除零概率
            entropy = -sum(prob .* log2(prob));
            totalEntropy = totalEntropy + entropy;
        end
        prevThresh = thresholds(i);
    end
    
    % 处理最后一个区域
    indices = binCenters > thresholds(end);
    if any(indices)
        prob = histCounts(indices) / sum(histCounts(indices));
        prob = prob(prob > 0);
        entropy = -sum(prob .* log2(prob));
        totalEntropy = totalEntropy + entropy;
    end
    
    % 由于ICOOT是最小化算法，返回负熵
    fitness = -totalEntropy;
end

function fitness = otsuObjective(thresholds, grayImage, numThresholds)
% Otsu目标函数（最大化类间方差）
    % 排序阈值
    thresholds = sort(thresholds);
    
    % 在[0,1]范围内直接处理
    totalPixels = numel(grayImage);
    
    % 创建[0,1]范围内的直方图（使用256个bin）
    binEdges = linspace(0, 1, 257);
    histCounts = histcounts(grayImage(:), binEdges);
    
    % 计算全局均值
    binCenters = linspace(0, 1, 256);
    globalMean = sum(binCenters .* histCounts) / totalPixels;
    
    % 计算类间方差
    betweenClassVariance = 0;
    prevThresh = -1;
    weightSum = 0;
    meanSum = 0;
    
    for i = 1:numThresholds
        % 计算当前类的权重和均值
        % 修复索引计算，确保是有效的正整数
        lowerBin = max(1, floor(prevThresh * 255) + 1); % 转换为bin索引，最小为1
        upperBin = min(256, floor(thresholds(i) * 255) + 1); % 最大为256
        
        if upperBin >= lowerBin && lowerBin <= 256 && upperBin <= 256
            currentHist = histCounts(lowerBin:upperBin);
            currentWeight = sum(currentHist) / totalPixels;
            if currentWeight > 0
                binValues = linspace(prevThresh, thresholds(i), length(currentHist));
                currentMean = sum(binValues .* currentHist) / sum(currentHist);
                betweenClassVariance = betweenClassVariance + currentWeight * (currentMean - globalMean)^2;
            end
        end
        prevThresh = thresholds(i);
    end
    
    % 处理最后一个类
    lowerBin = max(1, floor(prevThresh * 255) + 1); % 修复索引，最小为1
    if lowerBin <= 256
        currentHist = histCounts(lowerBin:end);
        currentWeight = sum(currentHist) / totalPixels;
        if currentWeight > 0
            binValues = linspace(prevThresh, 1, length(currentHist));
            currentMean = sum(binValues .* currentHist) / sum(currentHist);
            betweenClassVariance = betweenClassVariance + currentWeight * (currentMean - globalMean)^2;
        end
    end
    
    % 由于ICOOT是最小化算法，返回负的类间方差
    fitness = -betweenClassVariance;
end

function fitness = crossEntropyObjective(thresholds, grayImage, numThresholds)
% 交叉熵目标函数
    % 排序阈值
    thresholds = sort(thresholds);
    
    % 创建[0,1]范围内的直方图（使用256个bin）
    binEdges = linspace(0, 1, 257);
    histCounts = histcounts(grayImage(:), binEdges);
    histCounts = histCounts / sum(histCounts); % 归一化
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
    
    % 计算交叉熵
    crossEntropy = 0;
    prevThresh = 0;
    
    for i = 1:numThresholds
        % 找到当前阈值范围内的像素
        indices = binCenters > prevThresh & binCenters <= thresholds(i);
        if any(indices)
            prob = histCounts(indices) / sum(histCounts(indices));
            prob = prob(prob > 0); % 移除零概率
            % 理想分布假设为均匀分布
            idealProb = 1 / length(prob);
            entropy = -sum(prob .* log2(idealProb));
            crossEntropy = crossEntropy + entropy;
        end
        prevThresh = thresholds(i);
    end
    
    % 处理最后一个区域
    indices = binCenters > thresholds(end);
    if any(indices)
        prob = histCounts(indices) / sum(histCounts(indices));
        prob = prob(prob > 0);
        idealProb = 1 / length(prob);
        entropy = -sum(prob .* log2(idealProb));
        crossEntropy = crossEntropy + entropy;
    end
    
    % 交叉熵本身需要最小化
    fitness = crossEntropy;
end

function segmentedImage = applyThresholds(image, thresholds)
% 应用阈值进行图像分割
    % 确保阈值已排序
    thresholds = sort(thresholds);
    
    % 创建分割图像
    segmentedImage = zeros(size(image));
    
    % 应用阈值
    for i = 1:length(thresholds)
        if i == 1
            % 第一个区域
            segmentedImage(image <= thresholds(i)) = i;
        else
            % 中间区域
            segmentedImage(image > thresholds(i-1) & image <= thresholds(i)) = i;
        end
    end
    
    % 最后一个区域
    segmentedImage(image > thresholds(end)) = length(thresholds) + 1;
    
    % 归一化到0-1范围以便灰度显示
    segmentedImage = segmentedImage / (length(thresholds) + 1);

end

function edgeImage = generateEdgeStyleSegmentation(grayImage, thresholds)
% 生成边缘检测风格的分割结果（优化版）
    % 确保阈值已排序
    thresholds = sort(thresholds);
    
    % 首先应用阈值得到多区域分割
    segmentedImage = applyThresholds(grayImage, thresholds);
    
    % 1. 预处理：更轻微的高斯模糊，保留更多边缘细节
    [rows, cols] = size(segmentedImage);
    blurredImage = zeros(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            % 轻微模糊，保留更多边缘细节
            blurredImage(i,j) = (segmentedImage(i-1,j-1) + segmentedImage(i-1,j) + segmentedImage(i-1,j+1) + ...
                                segmentedImage(i,j-1) + 2*segmentedImage(i,j) + segmentedImage(i,j+1) + ...
                                segmentedImage(i+1,j-1) + segmentedImage(i+1,j) + segmentedImage(i+1,j+1)) / 10;
        end
    end
    
    % 2. 增强的Sobel边缘检测
    gradientMag = zeros(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            % 水平和垂直方向的梯度计算（更强的权重）
            Gx = 2*blurredImage(i-1,j+1) + 4*blurredImage(i,j+1) + 2*blurredImage(i+1,j+1) - ...
                 2*blurredImage(i-1,j-1) - 4*blurredImage(i,j-1) - 2*blurredImage(i+1,j-1);
            Gy = 2*blurredImage(i+1,j-1) + 4*blurredImage(i+1,j) + 2*blurredImage(i+1,j+1) - ...
                 2*blurredImage(i-1,j-1) - 4*blurredImage(i-1,j) - 2*blurredImage(i-1,j+1);
            
            % 计算梯度幅度
            gradientMag(i,j) = sqrt(Gx^2 + Gy^2);
        end
    end
    
    % 3. 降低阈值以检测更多边缘，使线条更完整
    maxGradient = max(gradientMag(:));
    meanGradient = mean(gradientMag(:));
    stdGradient = std(gradientMag(:));
    
    % 更敏感的阈值，检测更多边缘
    threshold = 0.15 * maxGradient + 0.1 * meanGradient - 0.05 * stdGradient;
    threshold = max(threshold, 0.02 * maxGradient); % 确保阈值不会太小
    
    % 应用阈值，创建二值边缘图像
    edgeImage = gradientMag > threshold;
    
    % 4. 修改边缘细化策略，保留更多边缘细节
    thinEdgeImage = zeros(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            if edgeImage(i,j) == 1
                % 更宽松的边缘保留条件
                neighbors = edgeImage(i-1:i+1, j-1:j+1);
                if sum(neighbors(:)) > 0 % 只要有一个邻接点就保留
                    thinEdgeImage(i,j) = 1;
                end
            end
        end
    end
    edgeImage = thinEdgeImage;
    
    % 反转边缘图像，使其成为白底黑线风格
    edgeImage = 1 - edgeImage;
    
    % 5. 更强的对比度增强，使线条更黑更清晰
    % 更强的非线性压缩，增强低灰度值（黑色）
    edgeImage = edgeImage .^ 0.4; % 更小的指数值，增强对比度
    edgeImage = max(0, min(1, edgeImage)); % 确保值在0-1之间
    
    % 6. 额外的后处理：强化边缘线条
    % 对边缘图像进行额外的对比度调整
    edgeImage = edgeImage * 1.1; % 提高整体亮度
    edgeImage(edgeImage < 0.2) = 0; % 抑制弱边缘
    edgeImage = max(0, min(1, edgeImage)); % 确保值在0-1之间
    
end
