% Lightning-Fast RF Fingerprinting - Extreme Speed Optimization
% Target: <2 minutes training, >98% accuracy, <0.1 loss
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

            fprintf('⚡ Lightning Processing: SNR=%d, Frames=%d, k=%d\n', localSNR, localFramesPerRouter, k);

            numKnownRouters = originalNumKnownRouters * k;
            numUnknownRouters = originalNumUnknownRouters * k;
            numTotalRouters = numKnownRouters + numUnknownRouters;
            SNR = localSNR;           % dB
            channelNumber = 1;        % WLAN channel number
            channelBand = 5;          % GHz
            frameLength = 160;        % L-LTF sequence length in samples
            san = 0.5;                % control the alpha

            numTotalFramesPerRouter = localFramesPerRouter;
            numTrainingFramesPerRouter = numTotalFramesPerRouter*0.8;
            numValidationFramesPerRouter = numTotalFramesPerRouter*0.1;
            numTestFramesPerRouter = numTotalFramesPerRouter*0.1;

            %% Generate alpha and beta parameters
            all_alpha = zeros(1,numTotalRouters*2);
            all_beta = zeros(1,numTotalRouters*2);

            for idx = 1:numTotalRouters
                alpha = generateAlpha(san); 
                beta = (alpha - 1) + 0.2 * rand(1) - 0.1; 
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

            % Configure multipath channel
            multipathChannel = comm.RayleighChannel(...
                'SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, ...
                'AveragePathGains', [0 -2 -10], ...
                'MaximumDopplerShift', 0);

            % Define RF impairment ranges
            phaseNoiseRange = [0.01, 0.3];
            freqOffsetRange = [-4, 4];
            dcOffsetRange = [-50, -32];

            % Set random seed for reproducibility
            rng(123456)  

            % Generate radio impairments for each router
            radioImpairments = repmat(...
                struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), ...
                numTotalRouters, 1);
            for routerIdx = 1:numTotalRouters
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
            validationIndices = 1:numValidationFramesPerRouter;
            testIndices = 1:numTestFramesPerRouter;

            tic
            generatedMACAddresses = strings(numTotalRouters, 1);

            %% Parallel data generation (optimized)
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
                        rxSig = awgn(rxImpairment,SNR,0);

                        [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSig);
                        LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                        if valid
                            frameCount=frameCount+1;
                            rxLLTF(:,frameCount) = LLTF;
                        end
                    end

                    % Randomize frame order
                    rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));

                    % Split data into training, validation, and test sets
                    idxStartTrain = (idx-1)*numTrainingFramesPerRouter + 1;
                    idxEndTrain = idx*numTrainingFramesPerRouter;
                    localxTrainingFrames(:, idxStartTrain:idxEndTrain) = rxLLTF(:, trainingIndices);

                    idxStartVal = (idx-1)*numValidationFramesPerRouter + 1;
                    idxEndVal = idx*numValidationFramesPerRouter;
                    localxValFrames(:, idxStartVal:idxEndVal) = rxLLTF(:, validationIndices+ numTrainingFramesPerRouter);

                    idxStartTest = (idx-1)*numTestFramesPerRouter + 1;
                    idxEndTest = idx*numTestFramesPerRouter;
                    localxTestFrames(:, idxStartTest:idxEndTest) = rxLLTF(:, testIndices + numTrainingFramesPerRouter+numValidationFramesPerRouter);
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
            fprintf('⚡ Data generation: %.1fs\n', GenerateTime);

            %% Prepare labels
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter);
            yVal = repelem(labels, numValidationFramesPerRouter);
            yTest = repelem(labels, numTestFramesPerRouter);

            %% Lightning-Fast Feature Engineering (Optimized)
            fprintf('⚡ Lightning feature engineering...\n');
            
            % Pre-compute complex data once
            xTrainComplex = complex(real(xTrainingFrames(:)), imag(xTrainingFrames(:)));
            xValComplex = complex(real(xValFrames(:)), imag(xValFrames(:)));
            xTestComplex = complex(real(xTestFrames(:)), imag(xTestFrames(:)));
            
            % Extract only the most discriminative features (reduced to 6 channels for speed)
            xTrainFeatures = [
                real(xTrainComplex), imag(xTrainComplex), ...              % Real, Imaginary
                abs(xTrainComplex), angle(xTrainComplex), ...              % Magnitude, Phase
                real(xTrainComplex).^2 + imag(xTrainComplex).^2, ...       % Power (combined)
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

            % Fast normalization (range is faster than zscore)
            xTrainFeatures = normalize(xTrainFeatures, 'range');
            xValFeatures = normalize(xValFeatures, 'range');
            xTestFeatures = normalize(xTestFeatures, 'range');

            % Reshape for CNN: [Height, Width, Channels, Samples] - 6 channels
            xTrainingFrames = permute(...
                reshape(xTrainFeatures,[frameLength,numTrainingFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);

            % Randomize training data
            vr = randperm(numTotalRouters*numTrainingFramesPerRouter);
            xTrainingFrames = xTrainingFrames(:,:,:,vr);
            yTrain = categorical(yTrain(vr));

            % Reshape validation and test data
            xValFrames = permute(...
                reshape(xValFeatures,[frameLength,numValidationFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);
            yVal = categorical(yVal);

            xTestFrames = permute(...
                reshape(xTestFeatures,[frameLength,numTestFramesPerRouter*numTotalRouters, 6, 1]),...
                [1 3 4 2]);
            yTest = categorical(yTest);

            %% Lightning-Fast Ultra-Compact Model
            fprintf('⚡ Building lightning-fast model...\n');
            inputSize = [frameLength 6 1];  % 6-channel for speed
            numClasses = numKnownRouters + 1;

            % Extremely streamlined architecture for maximum speed
            layers = [
                % Input
                imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input')  % No normalization for speed
                
                % Single powerful conv layer
                convolution2dLayer([11 6], 256, 'Stride', [3 1], 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                leakyReluLayer(0.1, 'Name', 'LReLU1')
                
                % Aggressive pooling for speed
                maxPooling2dLayer([4 1], 'Stride', [4 1], 'Name', 'MaxPool1')
                
                % Single ResNet-style block (minimal)
                convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'ResConv1')
                batchNormalizationLayer('Name', 'ResBN1')
                leakyReluLayer(0.1, 'Name', 'ResLReLU1')
                additionLayer(2, 'Name', 'ResAdd')
                
                % Final conv with extreme pooling
                convolution2dLayer([3 1], 512, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                leakyReluLayer(0.1, 'Name', 'LReLU2')
                
                % Ultra-aggressive global pooling
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % Minimal FC layers
                fullyConnectedLayer(256, 'Name', 'FC1')
                leakyReluLayer(0.1, 'Name', 'LRFC1')
                dropoutLayer(0.2, 'Name', 'Drop1')  % Minimal dropout
                
                % Direct to output
                fullyConnectedLayer(numClasses, 'Name', 'FCFinal')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ];

            % Create layer graph with single skip connection
            lgraph = layerGraph(layers);
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResAdd/in2');  % Skip connection

            % Lightning-fast training options
            miniBatchSize = 256;  % Maximum batch size for speed
            iterPerEpoch = ceil(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            options = trainingOptions('adam', ...
                'MaxEpochs', 8, ...               % Minimal epochs
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...  % Validate only once per epoch
                'Verbose', false, ...
                'InitialLearnRate', 0.02, ...     % Very high learning rate
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.2, ...
                'LearnRateDropPeriod', 3, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'none', ...              % No plots for speed
                'Shuffle', 'once', ...            % Shuffle only once
                'L2Regularization', 0.001, ...    % Minimal regularization
                'GradientThreshold', 1, ...
                'ValidationPatience', 2, ...      % Very early stopping
                'ExecutionEnvironment', 'cpu');

            % Lightning-fast training
            fprintf('⚡ Starting lightning training...\n');
            tic
            simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
            TrainTime = toc;
            fprintf('⚡ Training completed: %.1fs\n', TrainTime);

            %% Lightning evaluation
            fprintf('⚡ Lightning evaluation...\n');
            yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            testAccuracy = mean(yTest == yTestPred);
            
            % Fast loss calculation
            testProbs = predict(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            yTestOneHot = full(ind2vec(double(yTest)'))';
            testLoss = -mean(sum(yTestOneHot .* log(testProbs + 1e-8), 2));
            
            fprintf('⚡ Test Accuracy: %.2f%%\n', testAccuracy*100);
            fprintf('⚡ Test Loss: %.4f\n', testLoss);
            
            % Quick confusion matrix
            figure('Position', [100, 100, 800, 600]);
            cm = confusionchart(yTest, yTestPred);
            cm.Title = sprintf('Lightning-Fast Model\nSNR:%ddB, Frames:%d, Acc:%.2f%%, Loss:%.3f, Time:%.1fs', ...
                SNR, localFramesPerRouter, testAccuracy*100, testLoss, TrainTime);
            cm.RowSummary = 'row-normalized';
            
            confusionFileName = sprintf('Lightning_Confusion_%d_SNR_%d_Frame_%d', ...
                numTotalRouters, SNR, localFramesPerRouter);
            saveas(gcf, confusionFileName, 'png');

            %% Lightning statistical evaluation
            fprintf('⚡ Lightning statistics (10 tests)...\n');
            numTests = 10;  % Minimal tests for speed
            accuracies = zeros(numTests, 1);
            losses = zeros(numTests, 1);

            for i = 1:numTests
                % Ultra-fast subset testing
                idx = randperm(numel(yTest));
                subsetSize = min(500, numel(yTest));  % Very small subset
                xTestShuffled = xTestFrames(:,:,:,idx(1:subsetSize));
                yTestShuffled = yTest(idx(1:subsetSize));
                
                yPred = classify(simNet, xTestShuffled, 'ExecutionEnvironment', 'cpu');
                accuracies(i) = mean(yTestShuffled == yPred);
                
                % Fast loss calculation
                probs = predict(simNet, xTestShuffled, 'ExecutionEnvironment', 'cpu');
                yOneHot = full(ind2vec(double(yTestShuffled)'))';
                losses(i) = -mean(sum(yOneHot .* log(probs + 1e-8), 2));
            end

            % Final statistics
            avgAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            avgLoss = mean(losses);
            stdLoss = std(losses);
            
            fprintf('\n⚡ ======== LIGHTNING-FAST RESULTS ========\n');
            fprintf('⚡ Training Time:       %.1f seconds\n', TrainTime);
            fprintf('⚡ Data Generation:     %.1f seconds\n', GenerateTime);
            fprintf('⚡ Total Time:          %.1f seconds\n', TrainTime + GenerateTime);
            fprintf('🎯 Average Accuracy:    %.2f%% (±%.2f%%)\n', avgAccuracy*100, stdAccuracy*100);
            fprintf('📉 Average Loss:        %.4f (±%.4f)\n', avgLoss, stdLoss);
            fprintf('🔢 Model Features:      6-channel input, %d classes\n', numClasses);
            fprintf('💾 Total Routers:       %d (%d known + %d unknown)\n', ...
                numTotalRouters, numKnownRouters, numUnknownRouters);
            
            % Performance assessment
            if avgAccuracy >= 0.98 && avgLoss <= 0.1 && TrainTime <= 120
                fprintf('🏆 LIGHTNING SUCCESS: >98%% accuracy, <0.1 loss, <2min training!\n');
            elseif avgAccuracy >= 0.95 && avgLoss <= 0.2 && TrainTime <= 180
                fprintf('⚡ EXCELLENT: >95%% accuracy, <0.2 loss, <3min training!\n');
            elseif avgAccuracy >= 0.90 && TrainTime <= 300
                fprintf('👍 GOOD: >90%% accuracy, <5min training!\n');
            else
                fprintf('⚠️  Performance check: Acc=%.1f%%, Loss=%.3f, Time=%.1fs\n', ...
                    avgAccuracy*100, avgLoss, TrainTime);
            end
            fprintf('==========================================\n\n');

            %% Save lightning results
            resultsFileName = sprintf('Lightning_Results_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            
            save(resultsFileName, ...
                'avgAccuracy', 'stdAccuracy', 'avgLoss', 'stdLoss', ...
                'accuracies', 'losses', 'testAccuracy', 'testLoss', ...
                'GenerateTime', 'TrainTime', ...
                'numTotalRouters', 'numKnownRouters', 'numUnknownRouters', ...
                'SNR', 'localFramesPerRouter', 'numClasses', 'frameLength', ...
                'miniBatchSize', 'inputSize');
            
            fprintf('⚡ Results saved: %s\n', resultsFileName);
            
            % Save lightning network
            networkFileName = sprintf('Lightning_Network_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            save(networkFileName, 'simNet', 'lgraph', 'inputSize', 'numClasses', 'options');
            fprintf('⚡ Network saved: %s\n\n', networkFileName);
        end
    end
end

fprintf('⚡🏆 Lightning-fast processing completed!\n');

%% Lightning Helper Functions

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
    % Lightning-fast RF impairments
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
    % Lightning-fast phase noise
    try
        load('Mrms.mat','Mrms','MyI','xI');
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        % Ultra-fast fallback
        phaseNoise = -75 - 15*log10(abs(radioImpairments.FrequencyOffset) + 0.01) - ...
                     8*log10(radioImpairments.PhaseNoise + 0.001);
        phaseNoise = max(-120, min(-30, phaseNoise));
    end
end

function alpha = generateAlpha(san)
    % Lightning-fast alpha generation
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    alpha = max(1.2, min(2.8, alpha));  % Direct clamping
end