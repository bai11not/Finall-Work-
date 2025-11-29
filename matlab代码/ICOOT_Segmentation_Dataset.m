function ICOOT_Segmentation_Dataset(dataset_path, results_path, numThresholds, objFuncType, varargin)
% ICOOT_Segmentation_Dataset - 直接从数据集路径批量处理图像的ICOOT分割函数
% 'D:\jisuanfangfa\BSDS300\images\test\101085.jpg','D:\jisuanfangfa\matlab\ICOOT_results\BSD300_results',2,'otsu','MaxImages', 1, 'MaxIterations', 20(运行时输入此行命令即可运行单张图片处理，
修改D:\jisuanfangfa\BSDS300\images\test\101085.jpg图片路径可实现不同图片的分割情况)
% 输入参数:
%   dataset_path - 数据集路径，包含要处理的图像文件
%   results_path - 结果保存路径
%   numThresholds - 阈值数量
%   objFuncType - 目标函数类型: 'maxEntropy', 'otsu', 'crossEntropy'
%   varargin - 可选参数：
%     'PopulationSize' - 种群大小（默认50）
%     'MaxIterations' - 最大迭代次数（默认100）
%     'ImageType' - 图像类型过滤，如'*.jpg'（默认'*.jpg'）
%     'ShowProgress' - 是否显示进度（默认true）
%     'MaxImages' - 最大处理图像数量（默认0，表示处理所有图像）
%     'OutputEdgeStyle' - 是否输出边缘检测风格的结果（默认true）

    % 默认参数设置
    defaultParams = struct(...
        'PopulationSize', 50, ...
        'MaxIterations', 100, ...
        'ImageType', '*.jpg', ...
        'ShowProgress', true, ...
        'MaxImages', 0, ... % 0表示处理所有图像
        'OutputEdgeStyle',true); % 默认输出边缘检测风格结果
    
    % 解析输入参数
    p = inputParser;
    addParameter(p, 'PopulationSize', defaultParams.PopulationSize, @isnumeric);
    addParameter(p, 'MaxIterations', defaultParams.MaxIterations, @isnumeric);
    addParameter(p, 'ImageType', defaultParams.ImageType, @ischar);
    addParameter(p, 'ShowProgress', defaultParams.ShowProgress, @islogical);
    addParameter(p, 'MaxImages', defaultParams.MaxImages, @isnumeric);
    addParameter(p, 'OutputEdgeStyle', defaultParams.OutputEdgeStyle, @islogical);
    parse(p, varargin{:});
    params = p.Results;
    
    % 验证数据集路径（支持目录或单个文件）
    is_dir = exist(dataset_path, 'dir') == 7;
    is_file = exist(dataset_path, 'file') == 2;
    
    if ~is_dir && ~is_file
        error('路径不存在或不是有效的目录/文件: %s', dataset_path);
    end
    
    % 创建结果保存目录
    if ~exist(results_path, 'dir')
        mkdir(results_path);
        if params.ShowProgress
            disp(['创建结果目录: ' results_path]);
        end
    end
    
    % 获取图像文件列表
    if is_dir
        % 如果是目录，获取所有匹配的图像文件
        image_files = dir(fullfile(dataset_path, params.ImageType));
        total_images = length(image_files);
    else
        % 如果是单个文件，直接添加到列表
        [image_dir, image_name, image_ext] = fileparts(dataset_path);
        image_files = struct('name', [image_name image_ext], 'folder', image_dir);
        total_images = 1;
    end
    
    if total_images == 0
        warning('没有找到匹配的图像文件: %s', fullfile(dataset_path, params.ImageType));
        return;
    end
    
    % 限制处理图像数量
    if params.MaxImages > 0 && params.MaxImages < total_images
        total_images = params.MaxImages;
        if params.ShowProgress
            disp(['限制处理前' num2str(total_images) '张图像']);
        end
    end
    
    if params.ShowProgress
        disp(['开始处理数据集: ' dataset_path]);
        disp(['找到 ' num2str(length(image_files)) ' 张图像，将处理 ' num2str(total_images) ' 张']);
        disp(['阈值数量: ' num2str(numThresholds)]);
        disp(['目标函数: ' objFuncType]);
        disp('------------------------------');
    end
    
    % 初始化统计信息
    total_time = 0;
    all_thresholds = {};
    all_fitness = [];
    
    % 处理每张图像
    for i = 1:total_images
        if is_dir
            image_file = image_files(i);
            image_path = fullfile(dataset_path, image_file.name);
        else
            % 单个文件情况
            image_path = dataset_path;
        end
        [~, image_name, ~] = fileparts(image_path);
        
        if params.ShowProgress
            disp(['处理图像 ' num2str(i) '/' num2str(total_images) ': ' image_name]);
        end
        
        try
            % 读取图像
            originalImage = imread(image_path);
            
            % 预处理图像（灰度转换和归一化）
            if size(originalImage, 3) > 1
                grayImage = rgb2gray(originalImage);
            else
                grayImage = originalImage; % 如果已经是灰度图，直接使用
            end
            image = double(grayImage) / 255.0;
            
            % 添加调试信息
            if params.ShowProgress
                disp(['  图像尺寸: ' num2str(size(image))]);
                disp(['  图像范围: [' num2str(min(image(:))) ', ' num2str(max(image(:))) ']']);
                disp(['  阈值数量: ' num2str(numThresholds)]);
            end
            
            % 调用ICOOT_Segmentation进行分割
            tic;
            [segmented_image, best_thresholds, best_fitness, convergence_curve] = ICOOT_Segmentation(...
                originalImage, numThresholds, objFuncType, ...
                'PopulationSize', params.PopulationSize, ...
                'MaxIterations', params.MaxIterations, ...
                'ShowProgress', params.ShowProgress, ...
                'OutputEdgeStyle', params.OutputEdgeStyle); % 在子函数中显示进度和图像
            execution_time = toc;
            total_time = total_time + execution_time;
            
            % 保存分割图像
            result_image_path = fullfile(results_path, [image_name '_segmented.png']);
            imwrite(segmented_image, result_image_path);
            
            % 保存详细信息
            info_path = fullfile(results_path, [image_name '_info.txt']);
            saveImageInfo(info_path, image_name, numThresholds, objFuncType, ...
                best_thresholds, best_fitness, execution_time);
            
            % 保存原始数据（用于后续分析）
            mat_path = fullfile(results_path, [image_name '_data.mat']);
            save(mat_path, 'best_thresholds', 'best_fitness', 'convergence_curve', 'execution_time');
            
            % 存储统计信息
            all_thresholds{end+1} = best_thresholds; %#ok<AGROW>
            all_fitness(end+1) = best_fitness; %#ok<AGROW>
            
            if params.ShowProgress
                disp(['  完成! 最佳阈值: ' num2str(best_thresholds)]);
                disp(['  适应度值: ' num2str(best_fitness)]);
                disp(['  执行时间: ' num2str(execution_time, '%.4f') ' 秒']);
                disp(['  结果保存在: ' result_image_path]);
            end
            
        catch ME
            disp(['  处理图像时出错: ' ME.message]);
            error_log_path = fullfile(results_path, 'error_log.txt');
            fileID = fopen(error_log_path, 'a');
            fprintf(fileID, '%s: %s\n', image_name, ME.message);
            fclose(fileID);
        end
        
        if params.ShowProgress
            disp('------------------------------');
        end
    end
    
    % 生成汇总报告
    generateSummaryReport(results_path, numThresholds, objFuncType, ...
        total_images, total_time, all_thresholds, all_fitness);
    
    if params.ShowProgress
        disp('数据集处理完成!');
        disp(['平均处理时间: ' num2str(total_time/total_images, '%.4f') ' 秒/图像']);
        disp(['汇总报告保存在: ' fullfile(results_path, 'summary_report.txt')]);
    end
end

function saveImageInfo(info_path, image_name, num_thresholds, obj_func_type, ...
        thresholds, fitness, execution_time)
% 保存单个图像的处理信息
    fileID = fopen(info_path, 'w');
    fprintf(fileID, '图像名称: %s\n', image_name);
    fprintf(fileID, '阈值数量: %d\n', num_thresholds);
    fprintf(fileID, '目标函数: %s\n', obj_func_type);
    fprintf(fileID, '最佳阈值: ');
    for i = 1:length(thresholds)
        fprintf(fileID, '%.6f ', thresholds(i));
    end
    fprintf(fileID, '\n');
    fprintf(fileID, '最佳适应度: %.6f\n', fitness);
    fprintf(fileID, '执行时间: %.4f 秒\n', execution_time);
    fclose(fileID);
end

function generateSummaryReport(results_path, num_thresholds, obj_func_type, ...
        total_images, total_time, all_thresholds, all_fitness)
% 生成汇总报告
    report_path = fullfile(results_path, 'summary_report.txt');
    fileID = fopen(report_path, 'w');
    
    fprintf(fileID, 'ICOOT分割数据集汇总报告\n');
    fprintf(fileID, '=========================\n\n');
    fprintf(fileID, '处理时间: %s\n', datestr(now));
    fprintf(fileID, '阈值数量: %d\n', num_thresholds);
    fprintf(fileID, '目标函数: %s\n', obj_func_type);
    fprintf(fileID, '处理图像总数: %d\n', total_images);
    fprintf(fileID, '总处理时间: %.4f 秒\n', total_time);
    fprintf(fileID, '平均处理时间: %.4f 秒/图像\n\n', total_time/total_images);
    
    % 计算适应度统计
    fprintf(fileID, '适应度统计:\n');
    fprintf(fileID, '  最大值: %.6f\n', max(all_fitness));
    fprintf(fileID, '  最小值: %.6f\n', min(all_fitness));
    fprintf(fileID, '  平均值: %.6f\n', mean(all_fitness));
    fprintf(fileID, '  标准差: %.6f\n\n', std(all_fitness));
    
    % 阈值统计
    fprintf(fileID, '阈值统计:\n');
    if ~isempty(all_thresholds)
        for i = 1:num_thresholds
            threshold_values = cellfun(@(x) x(i), all_thresholds);
            fprintf(fileID, '  阈值 %d:\n', i);
            fprintf(fileID, '    平均值: %.6f\n', mean(threshold_values));
            fprintf(fileID, '    范围: [%.6f, %.6f]\n', min(threshold_values), max(threshold_values));
        end
    end
    
    fclose(fileID);
    
    % 保存所有数据到MAT文件以便后续分析
    mat_path = fullfile(results_path, 'all_results.mat');
    save(mat_path, 'num_thresholds', 'obj_func_type', 'total_images', ...
        'total_time', 'all_thresholds', 'all_fitness');
end
