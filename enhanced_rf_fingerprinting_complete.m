% Enhanced RF Fingerprinting with Hybrid CNN-ResNet-Attention-LSTM Model
% Compatible with MATLAB R2023b
% Fixed layer connections and optimized architecture

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

            fprintf('Processing SNR = %d, FramesPerRouter = %d, k = %d\n', localSNR, localFramesPerRouter, k);

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

            %% Parallel data generation
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
                    elapsedTime = seconds(toc);
                    elapsedTime.Format = 'hh:mm:ss';
                    fprintf('%s - Generating frames for router %d with MAC address %s on processor %d\n', ...
                        elapsedTime, routerIdx, localGeneratedMACAddresses(idx), spmdIndex);

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
            GenerateTime = seconds(toc);
            toc

            %% Prepare labels
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter);
            yVal = repelem(labels, numValidationFramesPerRouter);
            yTest = repelem(labels, numTestFramesPerRouter);

            %% Data preprocessing for deep learning
            xTrainingFrames = xTrainingFrames(:);
            xValFrames = xValFrames(:);
            xTestFrames = xTestFrames(:);

            % Separate real and imaginary parts
            xTrainingFrames = [real(xTrainingFrames), imag(xTrainingFrames)];
            xValFrames = [real(xValFrames), imag(xValFrames)];
            xTestFrames = [real(xTestFrames), imag(xTestFrames)];

            % Reshape for CNN input [Height, Width, Channels, Samples]
            xTrainingFrames = permute(...
                reshape(xTrainingFrames,[frameLength,numTrainingFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]);

            % Randomize training data
            vr = randperm(numTotalRouters*numTrainingFramesPerRouter);
            xTrainingFrames = xTrainingFrames(:,:,:,vr);
            yTrain = categorical(yTrain(vr));

            % Reshape validation data
            xValFrames = permute(...
                reshape(xValFrames,[frameLength,numValidationFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]);
            yVal = categorical(yVal);

            % Reshape test data
            xTestFrames = permute(...
                reshape(xTestFrames,[frameLength,numTestFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]);
            yTest = categorical(yTest);

            %% Enhanced Hybrid CNN-ResNet-Attention-LSTM Model
            inputSize = [frameLength 2 1]; 
            numClasses = numKnownRouters + 1;

            % Create layer graph with proper architecture
            lgraph = layerGraph();
            
            % Add input and initial CNN layers
            lgraph = addLayers(lgraph, [
                imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name', 'Input')
                convolution2dLayer([7 2], 64, 'Stride', [2 1], 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                maxPooling2dLayer([3 1], 'Stride', [2 1], 'Padding', 'same', 'Name', 'MaxPool1')
            ]);
            
            % Add ResNet Block 1 main path
            lgraph = addLayers(lgraph, [
                convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'ResConv1_1')
                batchNormalizationLayer('Name', 'ResBN1_1')
                reluLayer('Name', 'ResReLU1_1')
                convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'ResConv1_2')
                batchNormalizationLayer('Name', 'ResBN1_2')
            ]);
            
            % Add ResNet Block 1 addition and activation
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ResAdd1'));
            lgraph = addLayers(lgraph, reluLayer('Name', 'ResReLU1_Final'));
            
            % Add ResNet Block 2 main path
            lgraph = addLayers(lgraph, [
                convolution2dLayer([3 1], 128, 'Stride', [2 1], 'Padding', 'same', 'Name', 'ResConv2_1')
                batchNormalizationLayer('Name', 'ResBN2_1')
                reluLayer('Name', 'ResReLU2_1')
                convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'ResConv2_2')
                batchNormalizationLayer('Name', 'ResBN2_2')
            ]);
            
            % Add ResNet Block 2 skip connection
            lgraph = addLayers(lgraph, [
                convolution2dLayer([1 1], 128, 'Stride', [2 1], 'Name', 'ResSkip2')
                batchNormalizationLayer('Name', 'ResSkipBN2')
            ]);
            
            % Add ResNet Block 2 addition and activation
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ResAdd2'));
            lgraph = addLayers(lgraph, reluLayer('Name', 'ResReLU2_Final'));
            
            % Add feature processing and sequence layers
            lgraph = addLayers(lgraph, [
                globalAveragePooling2dLayer('Name', 'GAP')
                sequenceUnfoldingLayer('Name', 'SeqUnfold')
                flattenLayer('Name', 'Flatten')
                
                % Attention mechanism (simplified)
                fullyConnectedLayer(256, 'Name', 'AttentionFC')
                reluLayer('Name', 'AttentionReLU')
                dropoutLayer(0.2, 'Name', 'AttentionDropout')
                
                % LSTM layers for temporal modeling
                lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'LSTM1')
                dropoutLayer(0.3, 'Name', 'Dropout1')
                
                lstmLayer(64, 'OutputMode', 'last', 'Name', 'LSTM2')
                dropoutLayer(0.3, 'Name', 'Dropout2')
                
                % Final classification layers
                fullyConnectedLayer(128, 'Name', 'FC1')
                reluLayer('Name', 'FinalReLU')
                dropoutLayer(0.5, 'Name', 'FinalDropout')
                fullyConnectedLayer(numClasses, 'Name', 'FC2')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ]);
            
            % Connect all layers with proper skip connections
            % ResNet Block 1 connections
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResConv1_1');
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResAdd1/in2');  % Skip connection
            lgraph = connectLayers(lgraph, 'ResBN1_2', 'ResAdd1/in1');  % Main path
            lgraph = connectLayers(lgraph, 'ResAdd1', 'ResReLU1_Final');
            
            % ResNet Block 2 connections
            lgraph = connectLayers(lgraph, 'ResReLU1_Final', 'ResConv2_1');  % Main path
            lgraph = connectLayers(lgraph, 'ResReLU1_Final', 'ResSkip2');    % Skip path
            lgraph = connectLayers(lgraph, 'ResBN2_2', 'ResAdd2/in1');       % Main path
            lgraph = connectLayers(lgraph, 'ResSkipBN2', 'ResAdd2/in2');     % Skip path
            lgraph = connectLayers(lgraph, 'ResAdd2', 'ResReLU2_Final');
            
            % Final connection to feature processing
            lgraph = connectLayers(lgraph, 'ResReLU2_Final', 'GAP');

            % Training options with optimized hyperparameters
            miniBatchSize = 128;  % Optimized batch size
            iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            options = trainingOptions('adam', ...
                'MaxEpochs', 30, ...
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...
                'Verbose', false, ...
                'InitialLearnRate', 0.001, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...
                'LearnRateDropPeriod', 8, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.0001, ...
                'GradientThreshold', 1, ...
                'ValidationPatience', 5, ...
                'ExecutionEnvironment', 'cpu');

            % Train the enhanced model
            tic
            fprintf('Starting training of enhanced hybrid model...\n');
            simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
            TrainTime = seconds(toc);

            fprintf('Training completed in: ');
            toc

            %% Model evaluation
            fprintf('Evaluating model performance...\n');
            yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
            testAccuracy = mean(yTest == yTestPred);
            fprintf('Single Test Accuracy: %.2f%%\n', testAccuracy*100);
            
            % Create confusion matrix
            figure('Position', [100, 100, 800, 600]);
            cm = confusionchart(yTest, yTestPred);
            cm.Title = sprintf('Enhanced Model - SNR:%ddB, Frames:%d, Routers:%d', ...
                SNR, localFramesPerRouter, numTotalRouters);
            cm.RowSummary = 'row-normalized';
            cm.ColumnSummary = 'column-normalized';
            
            confusionFileName = sprintf('Enhanced_Confusion_%d_SNR_%d_Frame_%d', ...
                numTotalRouters, SNR, localFramesPerRouter);
            saveas(gcf, confusionFileName, 'png');
            fprintf('Confusion matrix saved as: %s.png\n', confusionFileName);

            %% Multiple test runs for statistical evaluation
            fprintf('Running multiple test evaluations for statistical analysis...\n');
            numTests = 50; % Reduced for faster execution
            accuracies = zeros(numTests, 1);

            for i = 1:numTests
                if mod(i, 10) == 0
                    fprintf('Test run %d/%d completed\n', i, numTests);
                end
                
                % Shuffle test data
                idx = randperm(numel(yTest));
                xTestFramesShuffled = xTestFrames(:,:,:,idx);
                yTestShuffled = yTest(idx);
                
                % Predict and calculate accuracy
                yTestPred = classify(simNet, xTestFramesShuffled, 'ExecutionEnvironment', 'cpu');
                accuracies(i) = mean(yTestShuffled == yTestPred);
            end

            % Calculate statistics
            averageAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            minAccuracy = min(accuracies);
            maxAccuracy = max(accuracies);
            
            fprintf('\n=== ENHANCED MODEL PERFORMANCE SUMMARY ===\n');
            fprintf('Average Accuracy: %.2f%% (±%.2f%%)\n', averageAccuracy*100, stdAccuracy*100);
            fprintf('Min Accuracy: %.2f%%\n', minAccuracy*100);
            fprintf('Max Accuracy: %.2f%%\n', maxAccuracy*100);
            fprintf('Data Generation Time: %.1f seconds\n', GenerateTime);
            fprintf('Training Time: %.1f seconds\n', TrainTime);
            fprintf('==========================================\n\n');

            %% Save comprehensive results
            saveFileName = sprintf('Enhanced_Complete_Result_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            
            save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', ...
                'stdAccuracy', 'minAccuracy', 'maxAccuracy', 'accuracies', ...
                'testAccuracy', 'numTotalRouters', 'SNR', 'localFramesPerRouter', ...
                'numClasses', 'frameLength', 'san');
            
            fprintf('Complete results saved to: %s\n\n', saveFileName);
            
            % Optional: Save the trained network
            networkFileName = sprintf('Enhanced_Network_%d_SNR_%d_Frame_%d.mat', ...
                numTotalRouters, SNR, localFramesPerRouter);
            save(networkFileName, 'simNet', 'inputSize', 'numClasses');
            fprintf('Trained network saved to: %s\n\n', networkFileName);
        end
    end
end

fprintf('All processing completed successfully!\n');

%% Helper Functions

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
% helperRFImpairments Apply RF impairments to signal
%   IMPAIREDSIG = helperRFImpairments(SIG, RADIOIMPAIRMENTS, FS) returns signal
%   SIG after applying the impairments defined by RADIOIMPAIRMENTS
%   structure at the sample rate FS.

    % Apply frequency offset
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset, ...
        'SampleRate', fs);

    % Apply phase noise
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise('Level', phaseNoise, ...
        'FrequencyOffset', abs(radioImpairments.FrequencyOffset));

    % Apply impairments in sequence
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);

    % Apply DC offset
    impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
% helperGetPhaseNoise Get phase noise value from lookup table
%   PHASENOISE = helperGetPhaseNoise(RADIOIMPAIRMENTS) returns the phase
%   noise value based on the radio impairments parameters.

    try
        load('Mrms.mat','Mrms','MyI','xI');
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        % Fallback if Mrms.mat is not available
        warning('Mrms.mat not found. Using simplified phase noise model.');
        phaseNoise = -80 - 20*log10(abs(radioImpairments.FrequencyOffset) + 1) - ...
            10*log10(radioImpairments.PhaseNoise);
    end
end

function alpha = generateAlpha(san)
% generateAlpha Generate alpha parameter with constraints
%   ALPHA = generateAlpha(SAN) generates an alpha value using normal
%   distribution with mean 1.5 and standard deviation SAN, constrained
%   to be between 1.2 and 2.8.

    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    
    % Ensure alpha is within the specified range
    maxAttempts = 100;  % Prevent infinite loop
    attempts = 0;
    while (alpha < 1.2 || alpha > 2.8) && attempts < maxAttempts
        alpha = mu + sigma * randn(1, 1);
        attempts = attempts + 1;
    end
    
    % Final bounds check
    alpha = max(1.2, min(2.8, alpha));
end