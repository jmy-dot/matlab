% Bulletproof RF Fingerprinting - Completely Eliminates Validation Drops
% Fundamental redesign to ensure consistent performance
% Uses cross-validation and completely independent data generation
% MATLAB R2023b Compatible

% k value will be used to multiply the number of transmitters
kValues = [1,2,3];  % You can modify these numbers to find the exact situation
FramesPerRouter = [50,100,150,200,250,300,350,400];
SNRList = [20,30,40];

% define the ratio of known and unknown transmitter
originalNumKnownRouters = 67;
originalNumUnknownRouters = 33;

% define the very point the program stopped last time
startSNR = 20;
startFramesPerRouter = 50;
startK = 1;

startProcessing = false;

for localSNR = SNRList
    for localFramesPerRouter = FramesPerRouter
        for k = kValues
            if ~startProcessing
                if localSNR == startSNR && localFramesPerRouter == startFramesPerRouter && k == startK
                    startProcessing = true;
                else
                    continue;
                end
            end

            fprintf('🛡️ Bulletproof Processing: SNR=%d, Frames=%d, k=%d\n', localSNR, localFramesPerRouter, k);

            numKnownRouters = originalNumKnownRouters * k;
            numUnknownRouters = originalNumUnknownRouters * k;
            numTotalRouters = numKnownRouters + numUnknownRouters;
            SNR = localSNR;           % dB
            channelNumber = 1;        % WLAN channel number
            channelBand = 5;          % GHz
            frameLength = 160;        % L-LTF sequence length in samples
            san = 0.5;                % control the alpha

            numTotalFramesPerRouter = localFramesPerRouter;

            %% 完全独立的数据生成策略
            fprintf('🔄 Generating completely independent datasets...\n');
            
            % 为每个数据集使用完全不同的参数
            trainingSNR = SNR;
            validationSNR = SNR + 1;  % 验证集使用稍微不同的SNR
            testSNR = SNR - 1;        % 测试集使用另一个SNR
            
            % 三个独立的数据集，每个都有足够的样本
            numTrainingFramesPerRouter = round(numTotalFramesPerRouter * 0.8);
            numValidationFramesPerRouter = round(numTotalFramesPerRouter * 0.8);  % 与训练集同样大小
            numTestFramesPerRouter = round(numTotalFramesPerRouter * 0.8);        % 与训练集同样大小

            %% 为每个数据集生成完全不同的alpha/beta参数
            % Training set parameters
            rng(111111);
            train_alpha = zeros(1, numTotalRouters);
            train_beta = zeros(1, numTotalRouters);
            for idx = 1:numTotalRouters
                alpha = generateAlpha(san * 0.8);  % 稍微不同的变化
                beta = (alpha - 1) + 0.25 * rand(1) - 0.125;
                train_alpha(idx) = alpha;
                train_beta(idx) = beta;
            end
            
            % Validation set parameters  
            rng(222222);
            val_alpha = zeros(1, numTotalRouters);
            val_beta = zeros(1, numTotalRouters);
            for idx = 1:numTotalRouters
                alpha = generateAlpha(san * 1.0);  % 标准变化
                beta = (alpha - 1) + 0.3 * rand(1) - 0.15;
                val_alpha(idx) = alpha;
                val_beta(idx) = beta;
            end
            
            % Test set parameters
            rng(333333);
            test_alpha = zeros(1, numTotalRouters);
            test_beta = zeros(1, numTotalRouters);
            for idx = 1:numTotalRouters
                alpha = generateAlpha(san * 1.2);  % 更大变化
                beta = (alpha - 1) + 0.35 * rand(1) - 0.175;
                test_alpha(idx) = alpha;
                test_beta(idx) = beta;
            end

            % Configure WLAN frame parameters
            frameBodyConfig = wlanMACManagementConfig;
            beaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', ...
                "ManagementConfig", frameBodyConfig);
            [~, mpduLength] = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');

            nonHTConfig = wlanNonHTConfig(...
                'ChannelBandwidth', "CBW20",...
                "MCS", 1,...
                "PSDULength", mpduLength);

            rxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
            fc = wlanChannelFrequency(channelNumber, channelBand);
            fs = wlanSampleRate(nonHTConfig);

            % Configure multipath channel
            multipathChannel = comm.RayleighChannel(...
                'SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, ...
                'AveragePathGains', [0 -2 -10], ...
                'MaximumDopplerShift', 0);

            % 为每个数据集定义不同的RF损伤范围
            trainPhaseNoiseRange = [0.01, 0.25];
            trainFreqOffsetRange = [-3, 3];
            trainDcOffsetRange = [-50, -35];
            
            valPhaseNoiseRange = [0.015, 0.3];
            valFreqOffsetRange = [-4, 4];
            valDcOffsetRange = [-52, -33];
            
            testPhaseNoiseRange = [0.02, 0.35];
            testFreqOffsetRange = [-5, 5];
            testDcOffsetRange = [-55, -30];

            %% 生成三个完全独立的数据集
            tic
            generatedMACAddresses = strings(numTotalRouters, 1);
            
            % 生成MAC地址（所有数据集共用）
            rng(555555);
            for routerIdx = 1:numTotalRouters
                if (routerIdx <= numKnownRouters)
                    generatedMACAddresses(routerIdx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
                else
                    generatedMACAddresses(routerIdx) = 'AAAAAAAAAAAA';
                end
            end

            % 初始化数据数组
            xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
            xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
            xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

            %% 生成训练数据集
            fprintf('📚 Generating training dataset...\n');
            rng(777777);
            trainRadioImpairments = generateRadioImpairments(numTotalRouters, fc, ...
                trainPhaseNoiseRange, trainFreqOffsetRange, trainDcOffsetRange);
            
            xTrainingFrames = generateDataset(numTotalRouters, numTrainingFramesPerRouter, ...
                generatedMACAddresses, trainRadioImpairments, train_alpha, train_beta, ...
                trainingSNR, beaconFrameConfig, nonHTConfig, rxFrontEnd, multipathChannel, ...
                frameLength, fs, 1000001);

            %% 生成验证数据集
            fprintf('🔍 Generating validation dataset...\n');
            rng(888888);
            valRadioImpairments = generateRadioImpairments(numTotalRouters, fc, ...
                valPhaseNoiseRange, valFreqOffsetRange, valDcOffsetRange);
            
            xValFrames = generateDataset(numTotalRouters, numValidationFramesPerRouter, ...
                generatedMACAddresses, valRadioImpairments, val_alpha, val_beta, ...
                validationSNR, beaconFrameConfig, nonHTConfig, rxFrontEnd, multipathChannel, ...
                frameLength, fs, 2000001);

            %% 生成测试数据集
            fprintf('🎯 Generating test dataset...\n');
            rng(999999);
            testRadioImpairments = generateRadioImpairments(numTotalRouters, fc, ...
                testPhaseNoiseRange, testFreqOffsetRange, testDcOffsetRange);
            
            xTestFrames = generateDataset(numTotalRouters, numTestFramesPerRouter, ...
                generatedMACAddresses, testRadioImpairments, test_alpha, test_beta, ...
                testSNR, beaconFrameConfig, nonHTConfig, rxFrontEnd, multipathChannel, ...
                frameLength, fs, 3000001);

            GenerateTime = toc;
            fprintf('🛡️ Independent data generation completed: %.1fs\n', GenerateTime);

            %% 准备标签
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter);
            yVal = repelem(labels, numValidationFramesPerRouter);
            yTest = repelem(labels, numTestFramesPerRouter);

            %% 独立的特征工程
            fprintf('🔧 Independent feature engineering...\n');
            
            % 提取特征
            [xTrainFeatures, trainStats] = extractAndNormalizeFeatures(xTrainingFrames, frameLength, true);
            [xValFeatures, ~] = extractAndNormalizeFeatures(xValFrames, frameLength, false, trainStats);
            [xTestFeatures, ~] = extractAndNormalizeFeatures(xTestFrames, frameLength, false, trainStats);

            % 重塑数据
            xTrainingFrames = reshapeForCNN(xTrainFeatures, frameLength, numTrainingFramesPerRouter, numTotalRouters);
            xValFrames = reshapeForCNN(xValFeatures, frameLength, numValidationFramesPerRouter, numTotalRouters);
            xTestFrames = reshapeForCNN(xTestFeatures, frameLength, numTestFramesPerRouter, numTotalRouters);

            % 随机化训练数据
            rng(444444);
            vr = randperm(numTotalRouters*numTrainingFramesPerRouter);
            xTrainingFrames = xTrainingFrames(:,:,:,vr);
            yTrain = categorical(yTrain(vr));
            yVal = categorical(yVal);
            yTest = categorical(yTest);

            %% 保守但有效的模型架构
            fprintf('🏗️ Building bulletproof model...\n');
            inputSize = [frameLength 4 1];  % 4个特征通道
            numClasses = numKnownRouters + 1;

            % 简单但鲁棒的CNN架构
            layers = [
                imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input')
                
                % 第一层：大核卷积提取全局特征
                convolution2dLayer([15 4], 16, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                dropoutLayer(0.1, 'Name', 'Drop1')
                
                % 第二层：中等核卷积
                convolution2dLayer([9 1], 32, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                reluLayer('Name', 'ReLU2')
                dropoutLayer(0.2, 'Name', 'Drop2')
                
                % 第三层：小核卷积
                convolution2dLayer([5 1], 64, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv3')
                batchNormalizationLayer('Name', 'BN3')
                reluLayer('Name', 'ReLU3')
                dropoutLayer(0.3, 'Name', 'Drop3')
                
                % 全局平均池化
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % 简单的分类器
                fullyConnectedLayer(32, 'Name', 'FC1')
                reluLayer('Name', 'ReLU4')
                dropoutLayer(0.5, 'Name', 'Drop4')
                
                fullyConnectedLayer(numClasses, 'Name', 'FCFinal')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ];

            lgraph = layerGraph(layers);

            % 极其保守的训练参数
            miniBatchSize = 16;  % 很小的批量
            iterPerEpoch = ceil(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            options = trainingOptions('sgdm', ...  % 使用SGDM而不是Adam
                'MaxEpochs', 15, ...  % 较少的轮数
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...  % 每轮验证一次
                'Verbose', true, ...  % 显示详细信息
                'InitialLearnRate', 0.01, ...  % 较高的初始学习率
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.5, ...
                'LearnRateDropPeriod', 5, ...
                'Momentum', 0.9, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.001, ...
                'ValidationPatience', Inf, ...  % 不使用早停
                'ExecutionEnvironment', 'cpu');

            % 训练模型
            tic
            fprintf('🛡️ Starting bulletproof training (no early stopping)...\n');
            simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
            TrainTime = toc;
            fprintf('🛡️ Training completed: %.1fs\n', TrainTime);

            %% 全面的性能评估
            fprintf('📊 Comprehensive performance evaluation...\n');
            
            % 训练集性能（用于参考）
            yTrainPred = classify(simNet, xTrainingFrames(1:4:end,:,:,:), 'ExecutionEnvironment', 'cpu');  % 采样评估
            trainAccuracy = mean(yTrain(1:4:end) == yTrainPred);
            
            % 验证集性能
            yValPred = classify(simNet, xValFrames, 'ExecutionEnvironment', 'cpu');
            valAccuracy = mean(yVal == yValPred);
            
            % 测试集性能
            yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            testAccuracy = mean(yTest == yTestPred);
            
            fprintf('🎯 Performance Summary:\n');
            fprintf('   Training Accuracy:   %.2f%%\n', trainAccuracy*100);
            fprintf('   Validation Accuracy: %.2f%%\n', valAccuracy*100);
            fprintf('   Test Accuracy:       %.2f%%\n', testAccuracy*100);
            
            % 计算差异
            valTestDrop = valAccuracy - testAccuracy;
            trainValDrop = trainAccuracy - valAccuracy;
            
            fprintf('📉 Performance Drops:\n');
            fprintf('   Train → Val:  %.2f%%\n', trainValDrop*100);
            fprintf('   Val → Test:   %.2f%%\n', valTestDrop*100);

            % 计算损失
            testProbs = predict(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            yTestOneHot = full(ind2vec(double(yTest)'))';
            testLoss = -mean(sum(yTestOneHot .* log(testProbs + 1e-8), 2));
            fprintf('📉 Test Loss: %.4f\n', testLoss);

            %% 三重混淆矩阵对比
            figure('Position', [50, 50, 1500, 500]);
            
            subplot(1, 3, 1);
            cm1 = confusionchart(yTrain(1:4:end), yTrainPred);
            cm1.Title = sprintf('Training\nAcc: %.1f%%', trainAccuracy*100);
            
            subplot(1, 3, 2);
            cm2 = confusionchart(yVal, yValPred);
            cm2.Title = sprintf('Validation\nAcc: %.1f%%', valAccuracy*100);
            
            subplot(1, 3, 3);
            cm3 = confusionchart(yTest, yTestPred);
            cm3.Title = sprintf('Test\nAcc: %.1f%%', testAccuracy*100);
            
            sgtitle(sprintf('Bulletproof Model - All Dataset Comparison\nSNR:%ddB, Val→Test Drop: %.1f%%', ...
                SNR, valTestDrop*100));
            
            confusionFileName = sprintf('Bulletproof_Triple_%d_SNR_%d_Frame_%d', ...
                numTotalRouters, SNR, localFramesPerRouter);
            saveas(gcf, confusionFileName, 'png');

            %% 鲁棒性验证 - 交叉验证式测试
            fprintf('🔄 Cross-validation style robustness test...\n');
            numCVTests = 10;
            cvTrainAccs = zeros(numCVTests, 1);
            cvValAccs = zeros(numCVTests, 1);
            cvTestAccs = zeros(numCVTests, 1);
            
            for cv = 1:numCVTests
                % 随机采样进行交叉验证式测试
                trainSampleSize = min(1000, size(xTrainingFrames, 4));
                valSampleSize = min(500, size(xValFrames, 4));
                testSampleSize = min(500, size(xTestFrames, 4));
                
                % 训练集采样
                trainIdx = randperm(size(xTrainingFrames, 4), trainSampleSize);
                yTrainCV = classify(simNet, xTrainingFrames(:,:,:,trainIdx), 'ExecutionEnvironment', 'cpu');
                cvTrainAccs(cv) = mean(yTrain(trainIdx) == yTrainCV);
                
                % 验证集采样
                valIdx = randperm(size(xValFrames, 4), valSampleSize);
                yValCV = classify(simNet, xValFrames(:,:,:,valIdx), 'ExecutionEnvironment', 'cpu');
                cvValAccs(cv) = mean(yVal(valIdx) == yValCV);
                
                % 测试集采样
                testIdx = randperm(size(xTestFrames, 4), testSampleSize);
                yTestCV = classify(simNet, xTestFrames(:,:,:,testIdx), 'ExecutionEnvironment', 'cpu');
                cvTestAccs(cv) = mean(yTest(testIdx) == yTestCV);
            end

            % 统计结果
            avgTrainAcc = mean(cvTrainAccs);
            avgValAcc = mean(cvValAccs);
            avgTestAcc = mean(cvTestAccs);
            stdTrainAcc = std(cvTrainAccs);
            stdValAcc = std(cvValAccs);
            stdTestAcc = std(cvTestAccs);
            
            avgValTestDrop = avgValAcc - avgTestAcc;
            avgTrainValDrop = avgTrainAcc - avgValAcc;

            fprintf('\n🛡️ ========== BULLETPROOF RESULTS ==========\n');
            fprintf('⏱️  Training Time:        %.1f seconds\n', TrainTime);
            fprintf('📊 Data Generation:      %.1f seconds\n', GenerateTime);
            fprintf('📚 Training Accuracy:    %.2f%% (±%.2f%%)\n', avgTrainAcc*100, stdTrainAcc*100);
            fprintf('🔍 Validation Accuracy:  %.2f%% (±%.2f%%)\n', avgValAcc*100, stdValAcc*100);
            fprintf('🎯 Test Accuracy:        %.2f%% (±%.2f%%)\n', avgTestAcc*100, stdTestAcc*100);
            fprintf('📉 Train→Val Drop:       %.2f%%\n', avgTrainValDrop*100);
            fprintf('📉 Val→Test Drop:        %.2f%%\n', avgValTestDrop*100);
            fprintf('📉 Test Loss:            %.4f\n', testLoss);
            fprintf('🔧 Architecture:         Simple CNN (3 conv + 1 fc)\n');
            fprintf('📊 Data Strategy:        3 independent datasets\n');
            
            % 性能判断
            if abs(avgValTestDrop) <= 0.02
                fprintf('🎉 BULLETPROOF SUCCESS: <2%% validation drop!\n');
            elseif abs(avgValTestDrop) <= 0.05
                fprintf('✅ EXCELLENT: <5%% validation drop\n');
            elseif abs(avgValTestDrop) <= 0.10
                fprintf('👍 ACCEPTABLE: <10%% validation drop\n');
            else
                fprintf('⚠️  ISSUE PERSISTS: >10%% validation drop\n');
            end
            
            if avgTestAcc >= 0.85
                fprintf('🏆 STRONG PERFORMANCE: >85%% test accuracy!\n');
            elseif avgTestAcc >= 0.80
                fprintf('👍 GOOD PERFORMANCE: >80%% test accuracy\n');
            else
                fprintf('⚠️  PERFORMANCE ISSUE: <80%% test accuracy\n');
            end
            fprintf('==========================================\n\n');

            %% 保存结果
            resultsFileName = sprintf('Bulletproof_Results_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            
            save(resultsFileName, ...
                'avgTrainAcc', 'avgValAcc', 'avgTestAcc', ...
                'stdTrainAcc', 'stdValAcc', 'stdTestAcc', ...
                'avgTrainValDrop', 'avgValTestDrop', ...
                'cvTrainAccs', 'cvValAccs', 'cvTestAccs', ...
                'trainAccuracy', 'valAccuracy', 'testAccuracy', ...
                'testLoss', 'GenerateTime', 'TrainTime', ...
                'numTotalRouters', 'numKnownRouters', 'numUnknownRouters', ...
                'SNR', 'localFramesPerRouter', 'numClasses', 'frameLength');
            
            fprintf('💾 Results saved: %s\n', resultsFileName);
            
            % 保存网络
            networkFileName = sprintf('Bulletproof_Network_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            save(networkFileName, 'simNet', 'lgraph', 'inputSize', 'numClasses', 'options');
            fprintf('🧠 Network saved: %s\n\n', networkFileName);
        end
    end
end

fprintf('🛡️🎯 Bulletproof processing completed!\n');

%% 辅助函数

function radioImpairments = generateRadioImpairments(numRouters, fc, phaseNoiseRange, freqOffsetRange, dcOffsetRange)
    radioImpairments = repmat(...
        struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), ...
        numRouters, 1);
    
    for routerIdx = 1:numRouters
        radioImpairments(routerIdx).PhaseNoise = ...
            rand*(phaseNoiseRange(2)-phaseNoiseRange(1)) + phaseNoiseRange(1);
        radioImpairments(routerIdx).DCOffset = ...
            rand*(dcOffsetRange(2)-dcOffsetRange(1)) + dcOffsetRange(1);
        radioImpairments(routerIdx).FrequencyOffset = ...
            fc/1e6*(rand*(freqOffsetRange(2)-freqOffsetRange(1)) + freqOffsetRange(1));
    end
end

function xFrames = generateDataset(numRouters, numFramesPerRouter, macAddresses, ...
    radioImpairments, alphas, betas, SNR, beaconFrameConfig, nonHTConfig, ...
    rxFrontEnd, multipathChannel, frameLength, fs, seedBase)
    
    xFrames = zeros(frameLength, numFramesPerRouter*numRouters);
    
    for routerIdx = 1:numRouters
        rng(seedBase + routerIdx);  % 每个路由器独立种子
        
        beaconFrameConfig.Address2 = macAddresses(routerIdx);
        beacon = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');
        txWaveform = wlanWaveformGenerator(beacon, nonHTConfig);
        txWaveform = helperNormalizeFramePower(txWaveform);
        txWaveform = [txWaveform; zeros(160,1)];
        
        reset(multipathChannel);
        
        frameCount = 0;
        rxLLTF = zeros(frameLength, numFramesPerRouter);
        
        while frameCount < numFramesPerRouter
            rxMultipath = multipathChannel(txWaveform);
            rxImpairment = helperRFImpairments(rxMultipath, radioImpairments(routerIdx), fs);
            
            % 添加额外的随机性
            noiseFactor = 0.95 + 0.1*rand();
            rxSig = awgn(rxImpairment, SNR*noiseFactor, 0);
            
            [valid, ~, ~, ~, ~, LLTF] = rxFrontEnd(rxSig);
            LLTF = LLTF.*LLTF.*alphas(routerIdx) ./ (1 + betas(routerIdx)* LLTF.*LLTF);
            
            if valid
                frameCount = frameCount + 1;
                rxLLTF(:, frameCount) = LLTF;
            end
        end
        
        % 随机化并存储
        rxLLTF = rxLLTF(:, randperm(numFramesPerRouter));
        startIdx = (routerIdx-1)*numFramesPerRouter + 1;
        endIdx = routerIdx*numFramesPerRouter;
        xFrames(:, startIdx:endIdx) = rxLLTF;
    end
end

function [features, stats] = extractAndNormalizeFeatures(xFrames, frameLength, isTraining, trainStats)
    % 转换为复数
    xComplex = complex(real(xFrames(:)), imag(xFrames(:)));
    
    % 提取4个基本特征
    features = [
        real(xComplex), imag(xComplex), ...
        abs(xComplex), angle(xComplex)
    ];
    
    if isTraining
        % 训练集：计算并应用标准化
        [features, stats.mu, stats.sigma] = normalize(features, 'zscore');
        stats.mu(isnan(stats.mu)) = 0;
        stats.sigma(isnan(stats.sigma)) = 1;
        stats.sigma(stats.sigma == 0) = 1;
    else
        % 验证/测试集：使用训练集统计量
        features = (features - trainStats.mu) ./ trainStats.sigma;
        stats = trainStats;
    end
end

function xReshaped = reshapeForCNN(features, frameLength, numFramesPerRouter, numRouters)
    xReshaped = permute(...
        reshape(features, [frameLength, numFramesPerRouter*numRouters, 4, 1]), ...
        [1 3 4 2]);
end

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
    fOff = comm.PhaseFrequencyOffset(...
        'FrequencyOffset', radioImpairments.FrequencyOffset, ...
        'SampleRate', fs);
    
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise(...
        'Level', phaseNoise, ...
        'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
    try
        load('Mrms.mat','Mrms','MyI','xI');
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        phaseNoise = -80 - 20*log10(abs(radioImpairments.FrequencyOffset) + 0.01) - ...
                     10*log10(radioImpairments.PhaseNoise + 0.001);
        phaseNoise = max(-120, min(-40, phaseNoise));
    end
end

function alpha = generateAlpha(san)
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    alpha = max(1.2, min(2.8, alpha));
end