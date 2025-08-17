% Robust RF Fingerprinting - Prevents Validation Set Overfitting
% Ensures consistent performance between training and testing
% Target: Stable 90-95% accuracy throughout training and testing
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

            fprintf('🔒 Robust Processing: SNR=%d, Frames=%d, k=%d\n', localSNR, localFramesPerRouter, k);

            numKnownRouters = originalNumKnownRouters * k;
            numUnknownRouters = originalNumUnknownRouters * k;
            numTotalRouters = numKnownRouters + numUnknownRouters;
            SNR = localSNR;           % dB
            channelNumber = 1;        % WLAN channel number
            channelBand = 5;          % GHz
            frameLength = 160;        % L-LTF sequence length in samples
            san = 0.5;                % control the alpha

            numTotalFramesPerRouter = localFramesPerRouter;
            % 改进数据分割比例，增加测试集，减少验证集
            numTrainingFramesPerRouter = round(numTotalFramesPerRouter*0.7);  % 70% training
            numValidationFramesPerRouter = round(numTotalFramesPerRouter*0.15); % 15% validation  
            numTestFramesPerRouter = numTotalFramesPerRouter - numTrainingFramesPerRouter - numValidationFramesPerRouter; % 15% test

            %% Generate alpha and beta parameters with more variation
            all_alpha = zeros(1,numTotalRouters*2);
            all_beta = zeros(1,numTotalRouters*2);

            for idx = 1:numTotalRouters
                alpha = generateAlpha(san); 
                beta = (alpha - 1) + 0.3 * rand(1) - 0.15;  % 增加beta变化范围
                all_alpha(idx)= alpha;
                all_beta(idx)= beta;
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

            % Configure multipath channel with more variation
            multipathChannel = comm.RayleighChannel(...
                'SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, ...
                'AveragePathGains', [0 -2 -10], ...
                'MaximumDopplerShift', 0);

            % Define RF impairment ranges with increased variation
            phaseNoiseRange = [0.005, 0.4];    % 扩大范围
            freqOffsetRange = [-5, 5];         % 扩大范围
            dcOffsetRange = [-55, -30];        % 扩大范围

            % 使用不同的随机种子确保数据多样性
            rng(123456 + localSNR + localFramesPerRouter + k);  

            % Generate radio impairments for each router with more variation
            radioImpairments = repmat(...
                struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), ...
                numTotalRouters, 1);
            for routerIdx = 1:numTotalRouters
                % 为每个路由器添加更多随机性
                rng(123456 + routerIdx + localSNR);
                radioImpairments(routerIdx).PhaseNoise = ...
                    rand*(phaseNoiseRange(2)-phaseNoiseRange(1)) + phaseNoiseRange(1);
                radioImpairments(routerIdx).DCOffset = ...
                    rand*(dcOffsetRange(2)-dcOffsetRange(1)) + dcOffsetRange(1);
                radioImpairments(routerIdx).FrequencyOffset = ...
                    fc/1e6*(rand*(freqOffsetRange(2)-freqOffsetRange(1)) + freqOffsetRange(1));
            end

            % Initialize data arrays
            xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
            xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
            xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

            trainingIndices = 1:numTrainingFramesPerRouter;
            validationIndices = (numTrainingFramesPerRouter+1):(numTrainingFramesPerRouter+numValidationFramesPerRouter);
            testIndices = (numTrainingFramesPerRouter+numValidationFramesPerRouter+1):numTotalFramesPerRouter;

            tic
            generatedMACAddresses = strings(numTotalRouters, 1);

            %% Robust data generation with strong randomization
            spmd
                routerIndices = spmdIndex:spmdSize:numTotalRouters;

                localGeneratedMACAddresses = strings(length(routerIndices), 1);
                localxTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*length(routerIndices));
                localxValFrames = zeros(frameLength, numValidationFramesPerRouter*length(routerIndices));
                localxTestFrames = zeros(frameLength, numTestFramesPerRouter*length(routerIndices));

                frameBodyConfig = wlanMACManagementConfig;
                localbeaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', ...
                    "ManagementConfig", frameBodyConfig);
                [~, mpduLength] = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
                localnonHTConfig = wlanNonHTConfig(...
                    'ChannelBandwidth', "CBW20",...
                    "MCS", 1,...
                    "PSDULength", mpduLength);
                localrxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
                localmultipathChannel = comm.RayleighChannel(...
                    'SampleRate', fs, ...
                    'PathDelays', [0 1.8 3.4]/fs, ...
                    'AveragePathGains', [0 -2 -10], ...
                    'MaximumDopplerShift', 0);

                localRadioImpairments = radioImpairments(routerIndices);
                local_all_alpha = all_alpha(routerIndices);
                local_all_beta = all_beta(routerIndices);

                for idx = 1:length(routerIndices)
                    routerIdx = routerIndices(idx);

                    % Generate MAC addresses
                    if (routerIdx<=numKnownRouters)
                        localGeneratedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
                    else
                        localGeneratedMACAddresses(idx) = 'AAAAAAAAAAAA';
                    end

                    localbeaconFrameConfig.Address2 = localGeneratedMACAddresses(idx);
                    beacon = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
                    txWaveform = wlanWaveformGenerator(beacon, localnonHTConfig);
                    txWaveform = helperNormalizeFramePower(txWaveform);
                    txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

                    reset(localmultipathChannel)

                    frameCount= 0;
                    rxLLTF = zeros(frameLength,numTotalFramesPerRouter);

                    while frameCount<numTotalFramesPerRouter
                        rxMultipath = localmultipathChannel(txWaveform);
                        rxImpairment = helperRFImpairments(rxMultipath, localRadioImpairments(idx), fs);
                        
                        % 增强噪声变化范围，确保训练/验证/测试数据的多样性
                        noiseFactor = 0.9 + 0.2*rand();  % 更大的SNR变化范围
                        rxSig = awgn(rxImpairment, SNR*noiseFactor, 0);

                        [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSig);
                        LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                        if valid
                            frameCount=frameCount+1;
                            rxLLTF(:,frameCount) = LLTF;
                        end
                    end

                    % 强化随机化，确保训练/验证/测试集真正独立
                    rng(456789 + routerIdx);  % 每个路由器独立的随机种子
                    rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));
                    
                    % 为不同数据集添加不同程度的噪声，增加真实性
                    baseNoiseLevel = 0.005 * std(rxLLTF(:));
                    
                    % Training data: 轻微噪声
                    trainData = rxLLTF(:, trainingIndices);
                    trainData = trainData + baseNoiseLevel * randn(size(trainData));
                    
                    % Validation data: 中等噪声  
                    valData = rxLLTF(:, validationIndices);
                    valData = valData + 1.5 * baseNoiseLevel * randn(size(valData));
                    
                    % Test data: 更多噪声，模拟真实环境
                    testData = rxLLTF(:, testIndices);
                    testData = testData + 2.0 * baseNoiseLevel * randn(size(testData));

                    % 分配到对应的数据集
                    idxStartTrain = (idx-1)*numTrainingFramesPerRouter + 1;
                    idxEndTrain = idx*numTrainingFramesPerRouter;
                    localxTrainingFrames(:, idxStartTrain:idxEndTrain) = trainData;

                    idxStartVal = (idx-1)*numValidationFramesPerRouter + 1;
                    idxEndVal = idx*numValidationFramesPerRouter;
                    localxValFrames(:, idxStartVal:idxEndVal) = valData;

                    idxStartTest = (idx-1)*numTestFramesPerRouter + 1;
                    idxEndTest = idx*numTestFramesPerRouter;
                    localxTestFrames(:, idxStartTest:idxEndTest) = testData;
                end

                generatedMACAddressesLab = localGeneratedMACAddresses;
                xTrainingFramesLab = localxTrainingFrames;
                xValFramesLab = localxValFrames;
                xTestFramesLab = localxTestFrames;
            end

            % Combine results from all workers
            generatedMACAddresses = vertcat(generatedMACAddressesLab{:});
            xTrainingFrames = horzcat(xTrainingFramesLab{:});
            xValFrames = horzcat(xValFramesLab{:});
            xTestFrames = horzcat(xTestFramesLab{:});
            GenerateTime = toc;
            fprintf('🔒 Robust data generation: %.1fs\n', GenerateTime);

            %% Prepare labels
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter);
            yVal = repelem(labels, numValidationFramesPerRouter);
            yTest = repelem(labels, numTestFramesPerRouter);

            %% Robust Feature Engineering
            fprintf('🔧 Robust feature engineering...\n');
            
            % Convert to complex
            xTrainComplex = complex(real(xTrainingFrames(:)), imag(xTrainingFrames(:)));
            xValComplex = complex(real(xValFrames(:)), imag(xValFrames(:)));
            xTestComplex = complex(real(xTestFrames(:)), imag(xTestFrames(:)));
            
            % 提取6个最稳定的特征（减少过拟合风险）
            xTrainFeatures = [
                real(xTrainComplex), imag(xTrainComplex), ...              % Real, Imaginary
                abs(xTrainComplex), angle(xTrainComplex), ...              % Magnitude, Phase
                real(xTrainComplex).^2 + imag(xTrainComplex).^2, ...       % Power
                unwrap(angle(xTrainComplex))                               % Unwrapped phase
            ];
            
            xValFeatures = [
                real(xValComplex), imag(xValComplex), ...
                abs(xValComplex), angle(xValComplex), ...
                real(xValComplex).^2 + imag(xValComplex).^2, ...
                unwrap(angle(xValComplex))
            ];
            
            xTestFeatures = [
                real(xTestComplex), imag(xTestComplex), ...
                abs(xTestComplex), angle(xTestComplex), ...
                real(xTestComplex).^2 + imag(xTestComplex).^2, ...
                unwrap(angle(xTestComplex))
            ];

            % 使用训练数据的统计量标准化所有数据集
            [xTrainFeatures, trainMu, trainSigma] = normalize(xTrainFeatures, 'zscore');
            xValFeatures = (xValFeatures - trainMu) ./ trainSigma;
            xTestFeatures = (xTestFeatures - trainMu) ./ trainSigma;

            % 只对训练数据添加特征噪声
            featureNoise = 0.05;  % 增加特征噪声
            xTrainFeatures = xTrainFeatures + featureNoise * randn(size(xTrainFeatures));

            % Reshape for CNN: [Height, Width, Channels, Samples] - 6 channels
            xTrainingFrames = permute(...
                reshape(xTrainFeatures,[frameLength,numTrainingFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);

            % 强化训练数据随机化
            rng(789012);
            vr = randperm(numTotalRouters*numTrainingFramesPerRouter);
            xTrainingFrames = xTrainingFrames(:,:,:,vr);
            yTrain = categorical(yTrain(vr));

            xValFrames = permute(...
                reshape(xValFeatures,[frameLength,numValidationFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);
            yVal = categorical(yVal);

            xTestFrames = permute(...
                reshape(xTestFeatures,[frameLength,numTestFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);
            yTest = categorical(yTest);

            %% Robust Model Architecture
            fprintf('🏗️ Building robust model...\n');
            inputSize = [frameLength 6 1];
            numClasses = numKnownRouters + 1;

            % 简化但更鲁棒的架构
            layers = [
                % Input
                imageInputLayer(inputSize, 'Normalization', 'zerocenter', 'Name', 'Input')
                
                % 第一个卷积块
                convolution2dLayer([9 6], 32, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                dropoutLayer(0.2, 'Name', 'Drop1')
                
                % 第二个卷积块
                convolution2dLayer([7 1], 64, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                reluLayer('Name', 'ReLU2')
                dropoutLayer(0.3, 'Name', 'Drop2')
                
                % 第三个卷积块
                convolution2dLayer([5 1], 128, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv3')
                batchNormalizationLayer('Name', 'BN3')
                reluLayer('Name', 'ReLU3')
                dropoutLayer(0.4, 'Name', 'Drop3')
                
                % 全局池化
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % 分类层
                fullyConnectedLayer(128, 'Name', 'FC1')
                batchNormalizationLayer('Name', 'BNFC1')
                reluLayer('Name', 'ReLUFC1')
                dropoutLayer(0.6, 'Name', 'Drop4')  % 强dropout
                
                fullyConnectedLayer(64, 'Name', 'FC2')
                batchNormalizationLayer('Name', 'BNFC2')
                reluLayer('Name', 'ReLUFC2')
                dropoutLayer(0.7, 'Name', 'Drop5')  % 超强dropout
                
                fullyConnectedLayer(numClasses, 'Name', 'FCFinal')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ];

            lgraph = layerGraph(layers);

            % 保守的训练参数，防止过拟合
            miniBatchSize = 32;  % 更小的批量
            iterPerEpoch = ceil(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            options = trainingOptions('adam', ...
                'MaxEpochs', 30, ...  % 更多轮次但更保守的学习
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', max(1, floor(iterPerEpoch/3)), ... % 更频繁的验证
                'Verbose', false, ...
                'InitialLearnRate', 0.001, ...  % 更保守的学习率
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...
                'LearnRateDropPeriod', 10, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.01, ...   % 强L2正则化
                'GradientThreshold', 0.5, ...   % 更严格的梯度裁剪
                'ValidationPatience', 5, ...    % 适中的patience
                'ExecutionEnvironment', 'cpu');

            % 训练鲁棒模型
            tic
            fprintf('🔒 Starting robust training...\n');
            simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
            TrainTime = toc;
            fprintf('🔒 Robust training completed: %.1fs\n', TrainTime);

            %% 分阶段评估 - 验证一致性
            fprintf('📊 Multi-stage evaluation for consistency check...\n');
            
            % 1. 验证集评估
            yValPred = classify(simNet, xValFrames, 'ExecutionEnvironment', 'cpu');
            valAccuracy = mean(yVal == yValPred);
            fprintf('🔍 Validation Accuracy: %.2f%%\n', valAccuracy*100);
            
            % 2. 测试集评估
            yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            testAccuracy = mean(yTest == yTestPred);
            fprintf('🔍 Test Accuracy: %.2f%%\n', testAccuracy*100);
            
            % 3. 一致性检查
            accuracyDrop = valAccuracy - testAccuracy;
            fprintf('📉 Accuracy Drop: %.2f%% (Val→Test)\n', accuracyDrop*100);
            
            if abs(accuracyDrop) > 0.05  % 5%以上的下降
                fprintf('⚠️  WARNING: Significant accuracy drop detected!\n');
            else
                fprintf('✅ CONSISTENT: Validation and test accuracy are consistent\n');
            end
            
            % 计算损失
            testProbs = predict(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            yTestOneHot = full(ind2vec(double(yTest)'))';
            testLoss = -mean(sum(yTestOneHot .* log(testProbs + 1e-8), 2));
            
            fprintf('📉 Test Loss: %.4f\n', testLoss);

            % 增强混淆矩阵
            figure('Position', [100, 100, 1200, 500]);
            
            subplot(1, 2, 1);
            cm1 = confusionchart(yVal, yValPred);
            cm1.Title = sprintf('Validation Set\nAccuracy: %.2f%%', valAccuracy*100);
            cm1.RowSummary = 'row-normalized';
            
            subplot(1, 2, 2);
            cm2 = confusionchart(yTest, yTestPred);
            cm2.Title = sprintf('Test Set\nAccuracy: %.2f%%', testAccuracy*100);
            cm2.RowSummary = 'row-normalized';
            
            sgtitle(sprintf('Robust Model Consistency Check\nSNR:%ddB, Frames:%d, Drop:%.1f%%', ...
                SNR, localFramesPerRouter, accuracyDrop*100));
            
            confusionFileName = sprintf('Robust_Consistency_%d_SNR_%d_Frame_%d', ...
                numTotalRouters, SNR, localFramesPerRouter);
            saveas(gcf, confusionFileName, 'png');

            %% 鲁棒性统计评估
            fprintf('🔒 Robust statistical evaluation...\n');
            numTests = 20;  % 减少测试次数，专注质量
            testAccuracies = zeros(numTests, 1);
            valAccuracies = zeros(numTests, 1);
            
            for i = 1:numTests
                % 测试集随机采样
                testIdx = randperm(numel(yTest));
                testSize = min(800, numel(yTest));
                xTestSample = xTestFrames(:,:,:,testIdx(1:testSize));
                yTestSample = yTest(testIdx(1:testSize));
                
                % 验证集随机采样
                valIdx = randperm(numel(yVal));
                valSize = min(400, numel(yVal));
                xValSample = xValFrames(:,:,:,valIdx(1:valSize));
                yValSample = yVal(valIdx(1:valSize));
                
                % 预测
                yTestPredSample = classify(simNet, xTestSample, 'ExecutionEnvironment', 'cpu');
                yValPredSample = classify(simNet, xValSample, 'ExecutionEnvironment', 'cpu');
                
                testAccuracies(i) = mean(yTestSample == yTestPredSample);
                valAccuracies(i) = mean(yValSample == yValPredSample);
            end

            % 统计分析
            avgTestAcc = mean(testAccuracies);
            stdTestAcc = std(testAccuracies);
            avgValAcc = mean(valAccuracies);
            stdValAcc = std(valAccuracies);
            avgDrop = avgValAcc - avgTestAcc;
            
            fprintf('\n🔒 ========== ROBUST MODEL RESULTS ==========\n');
            fprintf('⏱️  Training Time:       %.1f seconds\n', TrainTime);
            fprintf('📊 Data Generation:     %.1f seconds\n', GenerateTime);
            fprintf('🔍 Validation Accuracy: %.2f%% (±%.2f%%)\n', avgValAcc*100, stdValAcc*100);
            fprintf('🎯 Test Accuracy:       %.2f%% (±%.2f%%)\n', avgTestAcc*100, stdTestAcc*100);
            fprintf('📉 Average Drop:        %.2f%% (Val→Test)\n', avgDrop*100);
            fprintf('📉 Test Loss:           %.4f\n', testLoss);
            fprintf('🔢 Data Split:          %d%% train, %d%% val, %d%% test\n', ...
                round(100*numTrainingFramesPerRouter/numTotalFramesPerRouter), ...
                round(100*numValidationFramesPerRouter/numTotalFramesPerRouter), ...
                round(100*numTestFramesPerRouter/numTotalFramesPerRouter));
            
            % 性能评估
            if abs(avgDrop) <= 0.03  % 3%以内的下降
                fprintf('🎉 EXCELLENT CONSISTENCY: <3%% accuracy drop!\n');
            elseif abs(avgDrop) <= 0.05  % 5%以内的下降
                fprintf('✅ GOOD CONSISTENCY: <5%% accuracy drop\n');
            else
                fprintf('⚠️  NEEDS IMPROVEMENT: >5%% accuracy drop\n');
            end
            
            if avgTestAcc >= 0.90
                fprintf('🏆 HIGH PERFORMANCE: >90%% test accuracy achieved!\n');
            elseif avgTestAcc >= 0.85
                fprintf('👍 GOOD PERFORMANCE: >85%% test accuracy\n');
            else
                fprintf('⚠️  PERFORMANCE ISSUE: <85%% test accuracy\n');
            end
            fprintf('==========================================\n\n');

            %% 保存鲁棒结果
            resultsFileName = sprintf('Robust_Results_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            
            save(resultsFileName, ...
                'avgTestAcc', 'stdTestAcc', 'avgValAcc', 'stdValAcc', 'avgDrop', ...
                'testAccuracies', 'valAccuracies', 'testAccuracy', 'valAccuracy', ...
                'testLoss', 'GenerateTime', 'TrainTime', ...
                'numTotalRouters', 'numKnownRouters', 'numUnknownRouters', ...
                'SNR', 'localFramesPerRouter', 'numClasses', 'frameLength', ...
                'numTrainingFramesPerRouter', 'numValidationFramesPerRouter', 'numTestFramesPerRouter');
            
            fprintf('💾 Results saved: %s\n', resultsFileName);
            
            % 保存鲁棒网络
            networkFileName = sprintf('Robust_Network_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            save(networkFileName, 'simNet', 'lgraph', 'inputSize', 'numClasses', 'options');
            fprintf('🧠 Network saved: %s\n\n', networkFileName);
        end
    end
end

fprintf('🔒🎯 Robust processing with consistency validation completed!\n');

%% Enhanced Helper Functions

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
    % 增强的RF损伤建模，增加更多真实性
    fOff = comm.PhaseFrequencyOffset(...
        'FrequencyOffset', radioImpairments.FrequencyOffset, ...
        'SampleRate', fs);
    
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise(...
        'Level', phaseNoise, ...
        'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    
    % 增强DC偏移变化
    dcVariation = 1 + 0.1 * (rand() - 0.5);  % ±5% variation
    impairedSig = impPhNoise + dcVariation * 10^(radioImpairments.DCOffset/10);
    
    % 添加微小的非线性失真
    nonlinearFactor = 0.98 + 0.04 * rand();  % 轻微非线性
    impairedSig = impairedSig * nonlinearFactor;
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
    % 增强的相位噪声建模
    try
        load('Mrms.mat','Mrms','MyI','xI');
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        % 更真实的fallback模型
        phaseNoise = -75 - 20*log10(abs(radioImpairments.FrequencyOffset) + 0.001) - ...
                     15*log10(radioImpairments.PhaseNoise + 0.0001);
        % 添加随机变化
        phaseNoise = phaseNoise + 5 * (rand() - 0.5);
        phaseNoise = max(-130, min(-30, phaseNoise));
    end
end

function alpha = generateAlpha(san)
    % 增强的alpha生成，确保更多样性
    mu = 1.5;
    sigma = san;
    
    % 生成更多样化的alpha值
    maxAttempts = 100;
    for attempt = 1:maxAttempts
        alpha = mu + sigma * randn(1, 1);
        if alpha >= 1.2 && alpha <= 2.8
            % 添加更多变化
            alpha = alpha + 0.05 * (rand() - 0.5);
            alpha = max(1.2, min(2.8, alpha));
            return;
        end
    end
    
    % Fallback
    alpha = max(1.2, min(2.8, mu + sigma * randn(1, 1)));
    alpha = alpha + 0.05 * (rand() - 0.5);
    alpha = max(1.2, min(2.8, alpha));
end