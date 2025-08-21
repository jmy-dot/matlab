% Enhanced RF Fingerprinting with True ResNet Skip Connections + TCN + BiLSTM
% Compatible with MATLAB R2023b
% Target: >90% accuracy (where feasible), loss < 1, reasonable training time, reduce overfitting

%% Experiment sweep controls
kValues = [1,2,3];                  % Multiplier for number of transmitters
FramesPerRouter = [50,100,150,200]; % You can extend if needed
SNRList = [-10, -5, 0, 5, 10];

% Define ratio of known and unknown transmitters
originalNumKnownRouters = 67;
originalNumUnknownRouters = 33;

% Resume controls
startSNR = -10;
startFramesPerRouter = 50;
startK = 1;
startProcessing = false;

% Random seed for repeatability
rng(123456);

% Speed/robustness toggles
enableParallel = false;    % Set false to avoid parallel overhead and potential stalls
showTrainingPlot = true;   % Disable training-progress UI for speed

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

    % Problem configuration
    numKnownRouters = originalNumKnownRouters * k;
    numUnknownRouters = originalNumUnknownRouters * k;
    numTotalRouters = numKnownRouters + numUnknownRouters;
    SNR = localSNR;           % dB
    channelNumber = 1;        % WLAN channel number
    channelBand = 5;          % GHz
    frameLength = 160;        % L-LTF sequence length in samples
    san = 0.5;                % controls alpha distribution

    numTotalFramesPerRouter = localFramesPerRouter;
    numTrainingFramesPerRouter = floor(numTotalFramesPerRouter*0.8);
    numValidationFramesPerRouter = floor(numTotalFramesPerRouter*0.1);
    numTestFramesPerRouter = numTotalFramesPerRouter - numTrainingFramesPerRouter - numValidationFramesPerRouter;

    %% Per-router unique nonlinearity parameters
    all_alpha = zeros(1,numTotalRouters*2);
    all_beta = zeros(1,numTotalRouters*2);
    for idx = 1:numTotalRouters
        alpha = generateAlpha(san);
        beta = (alpha - 1) + 0.2 * rand(1) - 0.1;
        all_alpha(idx)= alpha;
        all_beta(idx)= beta;
    end

    % MAC/PHY configs
    frameBodyConfig = wlanMACManagementConfig;
    beaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', "ManagementConfig", frameBodyConfig);
    [~, mpduLength] = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');
    nonHTConfig = wlanNonHTConfig('ChannelBandwidth', "CBW20", "MCS", 1, "PSDULength", mpduLength);
    rxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
    fc = wlanChannelFrequency(channelNumber, channelBand);
    fs = wlanSampleRate(nonHTConfig);

    multipathChannel = comm.RayleighChannel('SampleRate', fs, ...
        'PathDelays', [0 1.8 3.4]/fs, 'AveragePathGains', [0 -2 -10], 'MaximumDopplerShift', 0);

    phaseNoiseRange = [0.01, 0.3];
    freqOffsetRange = [-4, 4];
    dcOffsetRange = [-50, -32];

    % Per-router RF impairments
    radioImpairments = repmat(struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), numTotalRouters, 1);
    for routerIdx = 1:numTotalRouters
        radioImpairments(routerIdx).PhaseNoise = rand*(phaseNoiseRange(2)-phaseNoiseRange(1)) + phaseNoiseRange(1);
        radioImpairments(routerIdx).DCOffset = rand*(dcOffsetRange(2)-dcOffsetRange(1)) + dcOffsetRange(1);
        radioImpairments(routerIdx).FrequencyOffset = fc/1e6*(rand*(freqOffsetRange(2)-freqOffsetRange(1)) + freqOffsetRange(1));
    end

    % Pre-alloc frame buffers (complex)
    xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
    xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
    xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

    trainingIndices = 1:numTrainingFramesPerRouter;
    validationIndices = 1:numValidationFramesPerRouter;
    testIndices = 1:numTestFramesPerRouter;

    tic
    generatedMACAddresses = strings(numTotalRouters, 1);

    % Parallel/serial frame generation
    if enableParallel && license('test','Distrib_Computing_Toolbox')
        spmd
            routerIndices = spmdIndex:spmdSize:numTotalRouters;

            localGeneratedMACAddresses = strings(length(routerIndices), 1);
            localxTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*length(routerIndices));
            localxValFrames = zeros(frameLength, numValidationFramesPerRouter*length(routerIndices));
            localxTestFrames = zeros(frameLength, numTestFramesPerRouter*length(routerIndices));

            frameBodyConfig = wlanMACManagementConfig;
            localbeaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', "ManagementConfig", frameBodyConfig);
            [~, mpduLength] = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
            localnonHTConfig = wlanNonHTConfig('ChannelBandwidth', "CBW20", "MCS", 1, "PSDULength", mpduLength);
            localrxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
            localmultipathChannel = comm.RayleighChannel('SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, 'AveragePathGains', [0 -2 -10], 'MaximumDopplerShift', 0);

            localRadioImpairments = radioImpairments(routerIndices);
            local_all_alpha = all_alpha(routerIndices);
            local_all_beta = all_beta(routerIndices);

            for idx = 1:length(routerIndices)
                routerIdx = routerIndices(idx);

                if (routerIdx<=numKnownRouters)
                    localGeneratedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
                else
                    localGeneratedMACAddresses(idx) = 'AAAAAAAAAAAA';
                end
                elapsedTime = seconds(toc); elapsedTime.Format = 'hh:mm:ss';
                fprintf('%s - Generating frames for router %d with MAC %s on worker %d\n', ...
                    elapsedTime, routerIdx, localGeneratedMACAddresses(idx), spmdIndex);

                localbeaconFrameConfig.Address2 = localGeneratedMACAddresses(idx);
                beacon = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
                txWaveform = wlanWaveformGenerator(beacon, localnonHTConfig);
                txWaveform = helperNormalizeFramePower(txWaveform);
                txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

                reset(localmultipathChannel)

                frameCount= 0;
                trials = 0; maxTrials = max(5*numTotalFramesPerRouter, 200);
                rxLLTF = zeros(frameLength,numTotalFramesPerRouter);

                while frameCount<numTotalFramesPerRouter && trials < maxTrials
                    trials = trials + 1;
                    rxMultipath = localmultipathChannel(txWaveform);
                    rxImpairment = helperRFImpairments(rxMultipath, localRadioImpairments(idx), fs);

                    % Fast path for low SNR: detect at high SNR, then degrade LLTF to target SNR
                    if SNR <= 0
                        rxSigFE = awgn(rxImpairment, 25, 'measured');
                    else
                        rxSigFE = awgn(rxImpairment, SNR, 'measured');
                    end

                    [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSigFE);

                    if valid
                        % Apply nonlinearity
                        LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                        % Degrade to target SNR if we used high-SNR detection
                        if SNR <= 0
                            Ps = mean(abs(LLTF).^2 + eps);
                            Nvar = Ps/10^(SNR/10);
                            noise = sqrt(Nvar/2) * (randn(size(LLTF)) + 1j*randn(size(LLTF)));
                            LLTF = LLTF + noise;
                        end

                        frameCount=frameCount+1;
                        rxLLTF(:,frameCount) = LLTF;
                    end
                end

                % If not enough frames collected, pad with last valid or zeros
                if frameCount < numTotalFramesPerRouter
                    if frameCount > 0
                        rxLLTF(:, frameCount+1:end) = repmat(rxLLTF(:,frameCount), 1, numTotalFramesPerRouter-frameCount);
                    else
                        rxLLTF(:, :) = 0;
                    end
                end

                % Randomize order
                rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));

                % Split into train/val/test buckets
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

        generatedMACAddresses = vertcat(generatedMACAddressesLab{:});
        xTrainingFrames = horzcat(xTrainingFramesLab{:});
        xValFrames = horzcat(xValFramesLab{:});
        xTestFrames = horzcat(xTestFramesLab{:});
    else
        routerIndices = 1:numTotalRouters;
        generatedMACAddresses = strings(numTotalRouters,1);
        xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
        xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
        xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

        % Local configs reused in loop
        frameBodyConfig = wlanMACManagementConfig;
        localbeaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', "ManagementConfig", frameBodyConfig);
        [~, mpduLength] = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
        localnonHTConfig = wlanNonHTConfig('ChannelBandwidth', "CBW20", "MCS", 1, "PSDULength", mpduLength);
        localrxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
        localmultipathChannel = comm.RayleighChannel('SampleRate', fs, 'PathDelays', [0 1.8 3.4]/fs, 'AveragePathGains', [0 -2 -10], 'MaximumDopplerShift', 0);
        localRadioImpairments = radioImpairments;
        local_all_alpha = all_alpha;
        local_all_beta = all_beta;

        for idx = 1:length(routerIndices)
            routerIdx = routerIndices(idx);
            if (routerIdx<=numKnownRouters)
                generatedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
            else
                generatedMACAddresses(idx) = 'AAAAAAAAAAAA';
            end

            localbeaconFrameConfig.Address2 = generatedMACAddresses(idx);
            beacon = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
            txWaveform = wlanWaveformGenerator(beacon, localnonHTConfig);
            txWaveform = helperNormalizeFramePower(txWaveform);
            txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

            reset(localmultipathChannel)

            frameCount= 0; trials = 0; maxTrials = max(5*numTotalFramesPerRouter, 200);
            rxLLTF = zeros(frameLength,numTotalFramesPerRouter);
            while frameCount<numTotalFramesPerRouter && trials < maxTrials
                trials = trials + 1;
                rxMultipath = localmultipathChannel(txWaveform);
                rxImpairment = helperRFImpairments(rxMultipath, localRadioImpairments(idx), fs);
                if SNR <= 0
                    rxSigFE = awgn(rxImpairment, 25, 'measured');
                else
                    rxSigFE = awgn(rxImpairment, SNR, 'measured');
                end
                [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSigFE);
                if valid
                    LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);
                    if SNR <= 0
                        Ps = mean(abs(LLTF).^2 + eps);
                        Nvar = Ps/10^(SNR/10);
                        noise = sqrt(Nvar/2) * (randn(size(LLTF)) + 1j*randn(size(LLTF)));
                        LLTF = LLTF + noise;
                    end
                    frameCount=frameCount+1;
                    rxLLTF(:,frameCount) = LLTF;
                end
            end
            if frameCount < numTotalFramesPerRouter
                if frameCount > 0
                    rxLLTF(:, frameCount+1:end) = repmat(rxLLTF(:,frameCount), 1, numTotalFramesPerRouter-frameCount);
                else
                    rxLLTF(:, :) = 0;
                end
            end

            rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));
            idxStartTrain = (idx-1)*numTrainingFramesPerRouter + 1; idxEndTrain = idx*numTrainingFramesPerRouter;
            xTrainingFrames(:, idxStartTrain:idxEndTrain) = rxLLTF(:, trainingIndices);
            idxStartVal = (idx-1)*numValidationFramesPerRouter + 1;   idxEndVal = idx*numValidationFramesPerRouter;
            xValFrames(:, idxStartVal:idxEndVal) = rxLLTF(:, validationIndices+ numTrainingFramesPerRouter);
            idxStartTest = (idx-1)*numTestFramesPerRouter + 1;        idxEndTest = idx*numTestFramesPerRouter;
            xTestFrames(:, idxStartTest:idxEndTest) = rxLLTF(:, testIndices + numTrainingFramesPerRouter+numValidationFramesPerRouter);
        end
    end
    GenerateTime = seconds(toc); %#ok<NASGU>
    toc

    %% Labels
    labels = generatedMACAddresses;
    labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";
    yTrain = repelem(labels, numTrainingFramesPerRouter);
    yVal = repelem(labels, numValidationFramesPerRouter);
    yTest = repelem(labels, numTestFramesPerRouter);

    %% Convert complex LLTF to robust I/Q features with derivatives, arrange as sequences
    numTrain = numel(yTrain);
    numVal = numel(yVal);
    numTest = numel(yTest);

    % Build cell arrays: each cell is [numFeatures x time]
    XTrain = cell(numTrain,1);
    for i = 1:numTrain
        Xi = xTrainingFrames(:, i); % complex [T x 1]
        Xf = extractFeaturesFromComplex(Xi); % [F x T]
        Xf = normalizePerSequence(Xf);
        % Stronger augmentation for low SNR
        Xf = augmentSequence(Xf, SNR);
        Xf = normalizePerSequence(Xf);
        XTrain{i} = Xf;
    end
    XVal = cell(numVal,1);
    for i = 1:numVal
        Xi = xValFrames(:, i);
        Xf = extractFeaturesFromComplex(Xi);
        Xf = normalizePerSequence(Xf);
        XVal{i} = Xf;
    end
    XTest = cell(numTest,1);
    for i = 1:numTest
        Xi = xTestFrames(:, i);
        Xf = extractFeaturesFromComplex(Xi);
        Xf = normalizePerSequence(Xf);
        XTest{i} = Xf;
    end

    % Shuffle training set
    vr = randperm(numTrain);
    XTrain = XTrain(vr);
    yTrain = categorical(yTrain(vr));
    yVal = categorical(yVal);
    yTest = categorical(yTest);

    % Compute class weights to mitigate imbalance (normalized around 1)
    classNames = categories(yTrain);
    counts = countcats(yTrain);
    invCounts = 1./max(counts,1);
    classWeights = invCounts / mean(invCounts);

    %% Build True ResNet + TCN + BiLSTM model (no unsupported attention)
    inputFeatureSize = 4;           % I, Q, dI, dQ per time step
    embedDim = 192;                 % Feature channels after CNN stack
    numClasses = numKnownRouters + 1; % include Unknown

    lgraph = layerGraph();
    [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize);

    % Optional denoise block for low SNR prior to stem
    [lgraph, lastName] = addDenoiseBlock1D(lgraph, inputName);
    % Initial 1D Conv stem
    [lgraph, lastName] = addStem1D(lgraph, lastName);

    % ResNet backbone
    [lgraph, lastName] = addResNetBackbone1D(lgraph, lastName, embedDim);

    % Extra temporal context via dilated residual blocks to mimic long-range modeling
    [lgraph, lastName] = addDilatedStack1D(lgraph, lastName, embedDim);

    % Channel alignment
    [lgraph, lastName] = addAlignBlock1D(lgraph, lastName, embedDim);

    % Replace attention with Temporal Convolutional Network stack
    [lgraph, lastName] = addTCNBlock1D(lgraph, lastName, embedDim);

    % BiLSTM stack
    [lgraph, lastName] = addBiLSTMStack(lgraph, lastName);

    % Classifier head
    [lgraph, lastName] = addClassifierHead(lgraph, lastName, numClasses, classNames, classWeights);

    %% Training options
    miniBatchSize = 128;
    iterPerEpoch = max(1, floor(numTrain/miniBatchSize));
    options = trainingOptions('adam', ...
        'MaxEpochs', 60, ...
        'ValidationData', {XVal, yVal}, ...
        'ValidationFrequency', iterPerEpoch, ...
        'ValidationPatience', 8, ...
        'Verbose', false, ...
        'InitialLearnRate', 2e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'L2Regularization', 2e-4, ...
        'GradientThreshold', 1, ...
        'Plots', ternary(showTrainingPlot,'training-progress','none'), ...
        'OutputNetwork', 'last-iteration', ...
        'ExecutionEnvironment', 'auto');

    %% Train
    tic
    fprintf('Training ResNet + TCN + BiLSTM model...\n');
    [simNet, trainInfo] = trainNetwork(XTrain, yTrain, lgraph, options); %#ok<ASGLU>
    TrainTime = seconds(toc); %#ok<NASGU>
    disp("Model training completed.");

    %% Evaluate (final-iteration model)
    yTestPred = classify(simNet, XTest, 'ExecutionEnvironment', 'auto');
    testAccuracy = mean(yTest == yTestPred);
    disp("Final model test accuracy: " + testAccuracy*100 + "%")
    figure
    cm = confusionchart(yTest, yTestPred);
    cm.Title = 'ResNet+TCN+BiLSTM Confusion Matrix (Test)';
    cm.RowSummary = 'row-normalized';
    confusionFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
    saveas(gcf,confusionFileName,'png');

    %% Statistical validation (multiple shuffles)
    numTests = 30; % keep runtime reasonable
    accuracies = zeros(numTests,1);
    for i = 1:numTests
        idx = randperm(numel(yTest));
        XTestShuffled = XTest(idx);
        yTestShuffled = yTest(idx);
        yTestPred = classify(simNet, XTestShuffled, 'ExecutionEnvironment', 'auto');
        accuracies(i) = mean(yTestShuffled == yTestPred);
    end
    averageAccuracy = mean(accuracies);
    stdAccuracy = std(accuracies);
    disp(['Average accuracy: ', num2str(averageAccuracy*100, '%.2f'), '% ± ', num2str(stdAccuracy*100, '%.2f'), '%']);

    %% Save
    saveFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
    save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'simNet');
    fprintf('Results saved to %s\n\n', saveFileName);
end
end
end


%% Helper: Residual block for 1-D sequences
function [lgraph, outName] = addResidualBlock1D(lgraph, blockName, inChannels, outChannels, stride, inputName)
% Adds a 1-D residual block with optional downsampling (via stride) and projection skip.
% Returns the updated lgraph and the output layer name of the block.

    mainLayers = [
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'Stride', stride, 'Name', blockName + "_conv1")
        batchNormalizationLayer('Name', blockName + "_bn1")
        reluLayer('Name', blockName + "_relu1")
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'Stride', 1, 'Name', blockName + "_conv2")
        batchNormalizationLayer('Name', blockName + "_bn2")
    ];
    skipNeeded = (inChannels ~= outChannels) || (stride ~= 1);
    if skipNeeded
        skipLayers = [
            convolution1dLayer(1, outChannels, 'Padding', 'same', 'Stride', stride, 'Name', blockName + "_skip_conv")
            batchNormalizationLayer('Name', blockName + "_skip_bn")
        ];
        lgraph = addLayers(lgraph, skipLayers);
    end
    lgraph = addLayers(lgraph, mainLayers);

    add = additionLayer(2, 'Name', blockName + "_add");
    lgraph = addLayers(lgraph, add);

    reluOut = reluLayer('Name', blockName + "_out");
    lgraph = addLayers(lgraph, reluOut);

    % Connections
    lgraph = connectLayers(lgraph, inputName, blockName + "_conv1");
    if skipNeeded
        lgraph = connectLayers(lgraph, inputName, blockName + "_skip_conv");
        lgraph = connectLayers(lgraph, blockName + "_skip_bn", blockName + "_add/in2");
    else
        lgraph = connectLayers(lgraph, inputName, blockName + "_add/in2");
    end
    lgraph = connectLayers(lgraph, blockName + "_bn2", blockName + "_add/in1");
    lgraph = connectLayers(lgraph, blockName + "_add", blockName + "_out");

    outName = blockName + "_out";
end

%% Helper: Dilated residual block for 1-D sequences
function [lgraph, outName] = addDilatedResidual1D(lgraph, blockName, inChannels, outChannels, dilation, inputName)
% Residual block with dilated conv to capture longer temporal context
    mainLayers = [
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'DilationFactor', dilation, 'Stride', 1, 'Name', blockName + "_conv1")
        batchNormalizationLayer('Name', blockName + "_bn1")
        reluLayer('Name', blockName + "_relu1")
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'DilationFactor', 1, 'Stride', 1, 'Name', blockName + "_conv2")
        batchNormalizationLayer('Name', blockName + "_bn2")
    ];
    skipNeeded = (inChannels ~= outChannels);
    if skipNeeded
        skipLayers = [
            convolution1dLayer(1, outChannels, 'Padding', 'same', 'Stride', 1, 'Name', blockName + "_skip_conv")
            batchNormalizationLayer('Name', blockName + "_skip_bn")
        ];
        lgraph = addLayers(lgraph, skipLayers);
    end
    lgraph = addLayers(lgraph, mainLayers);
    add = additionLayer(2, 'Name', blockName + "_add");
    lgraph = addLayers(lgraph, add);
    reluOut = reluLayer('Name', blockName + "_out");
    lgraph = addLayers(lgraph, reluOut);
    lgraph = connectLayers(lgraph, inputName, blockName + "_conv1");
    if skipNeeded
        lgraph = connectLayers(lgraph, inputName, blockName + "_skip_conv");
        lgraph = connectLayers(lgraph, blockName + "_skip_bn", blockName + "_add/in2");
    else
        lgraph = connectLayers(lgraph, inputName, blockName + "_add/in2");
    end
    lgraph = connectLayers(lgraph, blockName + "_bn2", blockName + "_add/in1");
    lgraph = connectLayers(lgraph, blockName + "_add", blockName + "_out");
    outName = blockName + "_out";
end

%% Modular builders
function [lgraph, outName] = addStem1D(lgraph, inputName)
% Feature stem: shallow conv to expand channel capacity and stabilize input
    stem = [
        convolution1dLayer(7, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'stem_conv')
        batchNormalizationLayer('Name', 'stem_bn')
        reluLayer('Name', 'stem_relu')
        dropoutLayer(0.15, 'Name', 'stem_drop')
    ];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, inputName, 'stem_conv');
    outName = 'stem_drop';
end

function [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize)
% Input layer for 1-D sequence features
    inputName = 'input';
    inLayer = sequenceInputLayer(inputFeatureSize, 'Name', inputName);
    lgraph = addLayers(lgraph, inLayer);
end

function [lgraph, outName] = addDenoiseBlock1D(lgraph, inName)
% Denoise block: lightweight temporal smoothing + learnable enhancement
    blk = [
        averagePooling1dLayer(3, 'Stride',1, 'Padding','same', 'Name','denoise_avg')
        convolution1dLayer(3, 8, 'Padding','same','Stride',1,'Name','denoise_conv')
        batchNormalizationLayer('Name','denoise_bn')
        reluLayer('Name','denoise_relu')
    ];
    % 1x1 conv to match channels for residual connection
    matchConv = [
        convolution1dLayer(1, 8, 'Padding','same','Stride',1,'Name','denoise_match')
        batchNormalizationLayer('Name','denoise_match_bn')
    ];
    addName = 'denoise_add';
    outRelu = reluLayer('Name','denoise_out');
    lgraph = addLayers(lgraph, blk);
    lgraph = addLayers(lgraph, matchConv);
    lgraph = addLayers(lgraph, additionLayer(2,'Name',addName));
    lgraph = addLayers(lgraph, outRelu);
    lgraph = connectLayers(lgraph, inName, 'denoise_avg');
    lgraph = connectLayers(lgraph, 'denoise_relu', [addName '/in1']);
    lgraph = connectLayers(lgraph, inName, 'denoise_match');
    lgraph = connectLayers(lgraph, 'denoise_match_bn', [addName '/in2']);
    lgraph = connectLayers(lgraph, addName, 'denoise_out');
    outName = 'denoise_out';
end

function [lgraph, outName] = addResNetBackbone1D(lgraph, inName, embedDim)
% ResNet backbone: channel progression 64->128->embedDim with projection skips
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res1', 64, 64, 1, inName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res2', 64, 128, 2, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res3', 128, 128, 1, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res4', 128, embedDim, 2, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res5', embedDim, embedDim, 1, outName);
end

function [lgraph, outName] = addDilatedStack1D(lgraph, inName, embedDim)
% Two dilated residual blocks for long-range temporal context
    [lgraph, outName] = addDilatedResidual1D(lgraph, 'dres1', embedDim, embedDim, 2, inName);
    [lgraph, outName] = addDilatedResidual1D(lgraph, 'dres2', embedDim, embedDim, 4, outName);
end

function [lgraph, outName] = addAlignBlock1D(lgraph, inName, embedDim)
% 1x1 Conv alignment to enforce exact embedDim channels
    alignBlock = [
        convolution1dLayer(1, embedDim, 'Padding', 'same', 'Stride', 1, 'Name', 'align_conv')
        batchNormalizationLayer('Name', 'align_bn')
        reluLayer('Name', 'align_relu')
    ];
    lgraph = addLayers(lgraph, alignBlock);
    lgraph = connectLayers(lgraph, inName, 'align_conv');
    outName = 'align_relu';
end

function [lgraph, outName] = addTCNBlock1D(lgraph, inName, embedDim)
% Temporal Convolutional Network stack (dilated residual convs)
    [lgraph, name1] = addTCNUnit(lgraph, 'tcn1', embedDim, [1 2], 0.1, inName);
    [lgraph, name2] = addTCNUnit(lgraph, 'tcn2', embedDim, [2 4], 0.1, name1);
    [lgraph, outName] = addTCNUnit(lgraph, 'tcn3', embedDim, [4 8], 0.1, name2);
end

function [lgraph, outName] = addTCNUnit(lgraph, unitName, channels, dilations, dropoutP, inName)
% One TCN unit: two convs with increasing dilations and a residual skip
    conv1 = convolution1dLayer(5, channels, 'Padding','same', 'DilationFactor', dilations(1), 'Name', unitName + "_conv1");
    bn1 = batchNormalizationLayer('Name', unitName + "_bn1");
    relu1 = reluLayer('Name', unitName + "_relu1");
    drop1 = dropoutLayer(dropoutP, 'Name', unitName + "_drop1");
    conv2 = convolution1dLayer(5, channels, 'Padding','same', 'DilationFactor', dilations(2), 'Name', unitName + "_conv2");
    bn2 = batchNormalizationLayer('Name', unitName + "_bn2");
    addL = additionLayer(2, 'Name', unitName + "_add");
    relu2 = reluLayer('Name', unitName + "_out");

    seq = [conv1; bn1; relu1; drop1; conv2; bn2];
    lgraph = addLayers(lgraph, seq);
    lgraph = addLayers(lgraph, addL);
    lgraph = addLayers(lgraph, relu2);

    lgraph = connectLayers(lgraph, inName, unitName + "_conv1");
    lgraph = connectLayers(lgraph, unitName + "_bn2", unitName + "_add/in1");
    lgraph = connectLayers(lgraph, inName, unitName + "_add/in2");
    lgraph = connectLayers(lgraph, unitName + "_add", unitName + "_out");
    outName = unitName + "_out";
end

function [lgraph, outName] = addBiLSTMStack(lgraph, inName)
% Two-layer BiLSTM stack for temporal modeling
    rnn = [
        bilstmLayer(192, 'OutputMode', 'sequence', 'Name', 'bilstm1')
        dropoutLayer(0.3, 'Name', 'rnn_drop1')
        bilstmLayer(128, 'OutputMode', 'last', 'Name', 'bilstm2')
        dropoutLayer(0.3, 'Name', 'rnn_drop2')
    ];
    lgraph = addLayers(lgraph, rnn);
    lgraph = connectLayers(lgraph, inName, 'bilstm1');
    outName = 'rnn_drop2';
end

function [lgraph, outName] = addClassifierHead(lgraph, inName, numClasses, classNames, classWeights)
% Dense head with class weights for imbalance mitigation
    head = [
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        dropoutLayer(0.4, 'Name', 'head_drop')
        fullyConnectedLayer(numClasses, 'Name', 'fc_final')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output', 'Classes', classNames, 'ClassWeights', classWeights')
    ];
    lgraph = addLayers(lgraph, head);
    lgraph = connectLayers(lgraph, inName, 'fc1');
    outName = 'output';
end

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
% helperRFImpairments Apply RF impairments
%   IMPAIREDSIG = helperRFImpairments(SIG, RADIOIMPAIRMENTS, FS) returns signal
%   SIG after applying the impairments defined by RADIOIMPAIRMENTS structure at sample rate FS.

    % Frequency offset
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);
    % Phase noise
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    % DC offset
    impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);
end

function out = ternary(cond, a, b)
% Simple inline ternary for options
    if cond, out = a; else, out = b; end
end

function y = helperNormalizeFramePower(x)
% Normalize a complex waveform to unit average power
    p = mean(abs(x).^2 + eps);
    y = x ./ sqrt(p);
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
% helperGetPhaseNoise Get phase noise value from LUT
    load('Mrms.mat','Mrms','MyI','xI');
    [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
    [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
    phaseNoise = -abs(MyI(iRms, iFreqOffset));
end

function alpha = generateAlpha(san)
% Generate alpha with controlled variance and bounds
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1);
    end
end

function Xf = extractFeaturesFromComplex(x)
% Build robust features [I; Q; dI; dQ] from complex time series x (T x 1)
    I = real(x(:)).';
    Q = imag(x(:)).';
    dI = [I(1), diff(I)];
    dQ = [Q(1), diff(Q)];
    Xf = [I; Q; dI; dQ];
end

function Z = normalizePerSequence(Z)
% Per-feature zero-mean and RMS normalization
    for r = 1:size(Z,1)
        mu = mean(Z(r,:));
        Z(r,:) = Z(r,:) - mu;
        rmsv = sqrt(mean(Z(r,:).^2) + 1e-8);
        Z(r,:) = Z(r,:) / rmsv;
    end
end

function Z = augmentSequence(Z, SNR)
% Data augmentation tailored for low SNR
    T = size(Z,2);
    % Probabilities (stronger for SNR<=0)
    if SNR <= 0
        pMask = 0.35; pRot = 0.55; pGain = 0.45; pShift = 0.35; pNoise = 0.55;
        noiseStd = 0.06;
    else
        pMask = 0.25; pRot = 0.45; pGain = 0.35; pShift = 0.25; pNoise = 0.45;
        noiseStd = 0.04;
    end

    % Temporal mask
    if rand < pMask && T > 24
        mlen = randi([8,20]);
        t0 = randi([1, max(1, T-mlen+1)]);
        Z(:, t0:min(T, t0+mlen-1)) = 0;
    end

    % Phase jitter (rotate I/Q and dI/dQ consistently)
    if rand < pRot
        theta = (pi/180) * (randn*6); % ~N(0,6deg)
        R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        iq = R * Z(1:2,:);
        diq = R * Z(3:4,:);
        Z(1:2,:) = iq;
        Z(3:4,:) = diq;
    end

    % Random gain
    if rand < pGain
        g = 10^(randn*0.02); % ~ +/-0.17dB
        Z = Z * g;
    end

    % Small circular time shift
    if rand < pShift && T > 1
        s = randi([-3,3]);
        if s ~= 0
            Z = circshift(Z, [0 s]);
        end
    end

    % Additive Gaussian noise (on all features)
    if rand < pNoise
        Z = Z + noiseStd*randn(size(Z));
    end
end

