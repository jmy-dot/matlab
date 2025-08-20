% Enhanced RF Fingerprinting with True ResNet Skip Connections + BiLSTM (No unsupported attention layers)
% Compatible with MATLAB R2023b
% Target: >90% accuracy, loss < 1, reasonable training time, reduce overfitting

%% Experiment sweep controls
kValues = [1,2,3];                  % Multiplier for number of transmitters
FramesPerRouter = [50,100,150,200]; % You can extend if needed
SNRList = [20,30,40];

% Define ratio of known and unknown transmitters
originalNumKnownRouters = 67;
originalNumUnknownRouters = 33;

% Resume controls
startSNR = 20;
startFramesPerRouter = 50;
startK = 1;
startProcessing = false;

% Random seed for repeatability
rng(123456);

% Ensure custom layer class is accessible on path
if exist('TemporalSelfAttentionLayer','class') ~= 8
    try
        addpath(fileparts(mfilename('fullpath')));
    catch
        % If running from base workspace without file context, assume current folder contains the class file
    end
end

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

    % Pre-alloc frame buffers
    xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
    xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
    xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

    trainingIndices = 1:numTrainingFramesPerRouter;
    validationIndices = 1:numValidationFramesPerRouter;
    testIndices = 1:numTestFramesPerRouter;

    tic
    generatedMACAddresses = strings(numTotalRouters, 1);

    % Parallel frame generation
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
            rxLLTF = zeros(frameLength,numTotalFramesPerRouter);

            while frameCount<numTotalFramesPerRouter
                rxMultipath = localmultipathChannel(txWaveform);
                rxImpairment = helperRFImpairments(rxMultipath, localRadioImpairments(idx), fs);
                rxSig = awgn(rxImpairment,SNR,0);

                [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSig);

                % Nonlinear PA-like distortion (alpha/beta)
                LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                if valid
                    frameCount=frameCount+1;
                    rxLLTF(:,frameCount) = LLTF;
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
    GenerateTime = seconds(toc); %#ok<NASGU>
    toc

    %% Labels
    labels = generatedMACAddresses;
    labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";
    yTrain = repelem(labels, numTrainingFramesPerRouter);
    yVal = repelem(labels, numValidationFramesPerRouter);
    yTest = repelem(labels, numTestFramesPerRouter);

    %% Convert complex LLTF to I/Q features and arrange as sequences
    % x*Frames are [frameLength x numSamples], complex
    xTrainingFrames = [real(xTrainingFrames), imag(xTrainingFrames)];
    xValFrames = [real(xValFrames), imag(xValFrames)];
    xTestFrames = [real(xTestFrames), imag(xTestFrames)];

    % Reshape to [frameLength x 2 x N]
    numTrain = numel(yTrain);
    numVal = numel(yVal);
    numTest = numel(yTest);

    xTrainingFrames = reshape(xTrainingFrames, [frameLength, 2, numTrain]);
    xValFrames = reshape(xValFrames, [frameLength, 2, numVal]);
    xTestFrames = reshape(xTestFrames, [frameLength, 2, numTest]);

    % Shuffle training set
    vr = randperm(numTrain);
    xTrainingFrames = xTrainingFrames(:,:,vr);
    yTrain = categorical(yTrain(vr));
    yVal = categorical(yVal);
    yTest = categorical(yTest);

    % Compute class weights to mitigate imbalance (normalized around 1)
    classNames = categories(yTrain);
    counts = countcats(yTrain);
    invCounts = 1./max(counts,1);
    classWeights = invCounts / mean(invCounts);

    % Convert to cell arrays for sequence networks: each cell [features(=2) x time(=frameLength)]
    XTrain = cell(numTrain,1);
    for i = 1:numTrain
        Xi = squeeze(xTrainingFrames(:, :, i));     % [frameLength x 2]
        XTrain{i} = Xi.';                            % [2 x frameLength]
    end
    XVal = cell(numVal,1);
    for i = 1:numVal
        Xi = squeeze(xValFrames(:, :, i));
        XVal{i} = Xi.';
    end
    XTest = cell(numTest,1);
    for i = 1:numTest
        Xi = squeeze(xTestFrames(:, :, i));
        XTest{i} = Xi.';
    end

    % Lightweight training-time augmentation: temporal masking and small IQ rotation
    % Apply only to training set (probabilities kept conservative)
    for i = 1:numTrain
        Xi = XTrain{i}; % [2 x T]
        Tlen = size(Xi,2);
        % Temporal mask (SpecAugment-like)
        if rand < 0.15 && Tlen > 24
            mlen = randi([8,20]);
            t0 = randi([1, max(1, Tlen-mlen+1)]);
            Xi(:, t0:min(Tlen, t0+mlen-1)) = 0;
        end
        % Small random phase rotation
        if rand < 0.30
            theta = (pi/180) * (randn*3); % ~N(0,3deg)
            R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            Xi = R * Xi;
        end
        XTrain{i} = Xi;
    end

    %% Build True ResNet + BiLSTM model (no unsupported attention)
    inputFeatureSize = 2;           % I and Q per time step
    embedDim = 256;                 % Feature channels after CNN stack
    numClasses = numKnownRouters + 1; % include Unknown

    lgraph = layerGraph();
    [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize);

    % Initial 1D Conv stem
    [lgraph, lastName] = addStem1D(lgraph, inputName);

    % ResNet backbone
    [lgraph, lastName] = addResNetBackbone1D(lgraph, lastName, embedDim);

    % Extra temporal context via dilated residual blocks to mimic long-range modeling
    [lgraph, lastName] = addDilatedStack1D(lgraph, lastName, embedDim);

    % Channel alignment
    [lgraph, lastName] = addAlignBlock1D(lgraph, lastName, embedDim);

    % Inception-style multi-branch temporal blocks to increase connectivity
    [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc1');
    [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc2');

    % Temporal self-attention block
    [lgraph, lastName] = addSelfAttentionBlock(lgraph, lastName, embedDim);

    % BiLSTM stack
    % BiLSTM stack
    [lgraph, lastName] = addBiLSTMStack(lgraph, lastName);

    % Classifier head
    % Classifier head
    [lgraph, lastName] = addClassifierHead(lgraph, lastName, numClasses, classNames, classWeights);

    %% Training options
    miniBatchSize = 96;
    iterPerEpoch = max(1, floor(numTrain/miniBatchSize));
    options = trainingOptions('adam', ...
        'MaxEpochs', 45, ...
        'ValidationData', {XVal, yVal}, ...
        'ValidationFrequency', iterPerEpoch, ...
        'Verbose', false, ...
        'InitialLearnRate', 5e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'L2Regularization', 5e-5, ...
        'GradientThreshold', 1, ...
        'Plots', 'training-progress', ...
        'OutputNetwork', 'last-iteration', ...
        'ExecutionEnvironment', 'auto');

    %% Train
    tic
    fprintf('Training ResNet + BiLSTM model...\n');
    [simNet, trainInfo] = trainNetwork(XTrain, yTrain, lgraph, options); %#ok<ASGLU>
    TrainTime = seconds(toc); %#ok<NASGU>
    disp("Model training completed.");

    %% Evaluate (final-iteration model)
    yTestPred = classify(simNet, XTest, 'ExecutionEnvironment', 'auto');
    testAccuracy = mean(yTest == yTestPred);
    disp("Final model test accuracy: " + testAccuracy*100 + "%")
    figure
    cm = confusionchart(yTest, yTestPred);
    cm.Title = 'ResNet+BiLSTM Confusion Matrix (Test)';
    cm.RowSummary = 'row-normalized';
    confusionFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
    saveas(gcf,confusionFileName,'png');

    %% Statistical validation (multiple shuffles)
    numTests = 50; % fewer to keep runtime reasonable
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
        dropoutLayer(0.1, 'Name', 'stem_drop')
    ];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, inputName, 'stem_conv');
    outName = 'stem_drop';
end

function [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize)
% Input layer for 1-D sequence features (e.g., 2 for I/Q)
    inputName = 'input';
    inLayer = sequenceInputLayer(inputFeatureSize, 'Name', inputName);
    lgraph = addLayers(lgraph, inLayer);
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
% 1x1 Conv alignment to enforce exact embedDim channels before attention
    alignBlock = [
        convolution1dLayer(1, embedDim, 'Padding', 'same', 'Stride', 1, 'Name', 'align_conv')
        batchNormalizationLayer('Name', 'align_bn')
        reluLayer('Name', 'align_relu')
    ];
    lgraph = addLayers(lgraph, alignBlock);
    lgraph = connectLayers(lgraph, inName, 'align_conv');
    outName = 'align_relu';
end

function [lgraph, outName] = addSelfAttentionBlock(lgraph, inName, embedDim)
% LayerNorm + custom self-attention + dropout
    attn = [
        layerNormalizationLayer('Name', 'pre_attn_norm')
        TemporalSelfAttentionLayer(embedDim, 'self_attn')
        dropoutLayer(0.1, 'Name', 'attn_drop')
    ];
    lgraph = addLayers(lgraph, attn);
    lgraph = connectLayers(lgraph, inName, 'pre_attn_norm');
    outName = 'attn_drop';
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

function [lgraph, outName] = addInceptionDilated1D(lgraph, inName, outChannels, blockId)
% Inception-like parallel dilated temporal convs + residual aggregation
    br1 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br1_conv1'])
        reluLayer('Name',[blockId '_br1_relu1'])
        convolution1dLayer(3, outChannels/4, 'Padding','same','DilationFactor',1,'Name',[blockId '_br1_conv3'])
        batchNormalizationLayer('Name',[blockId '_br1_bn'])
        reluLayer('Name',[blockId '_br1_relu2'])
    ];
    br2 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br2_conv1'])
        reluLayer('Name',[blockId '_br2_relu1'])
        convolution1dLayer(3, outChannels/4, 'Padding','same','DilationFactor',2,'Name',[blockId '_br2_conv3d2'])
        batchNormalizationLayer('Name',[blockId '_br2_bn'])
        reluLayer('Name',[blockId '_br2_relu2'])
    ];
    br3 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br3_conv1'])
        reluLayer('Name',[blockId '_br3_relu1'])
        convolution1dLayer(5, outChannels/4, 'Padding','same','DilationFactor',3,'Name',[blockId '_br3_conv5d3'])
        batchNormalizationLayer('Name',[blockId '_br3_bn'])
        reluLayer('Name',[blockId '_br3_relu2'])
    ];
    br4 = [
        maxPooling1dLayer(3,'Stride',1,'Padding','same','Name',[blockId '_br4_pool'])
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br4_conv1'])
        batchNormalizationLayer('Name',[blockId '_br4_bn'])
        reluLayer('Name',[blockId '_br4_relu'])
    ];

    lgraph = addLayers(lgraph, br1);
    lgraph = addLayers(lgraph, br2);
    lgraph = addLayers(lgraph, br3);
    lgraph = addLayers(lgraph, br4);

    % Connect input to four branches
    lgraph = connectLayers(lgraph, inName, [blockId '_br1_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br2_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br3_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br4_pool']);

    % Concatenate branches
    concatName = [blockId '_concat'];
    lgraph = addLayers(lgraph, depthConcatenationLayer(4,'Name',concatName));
    lgraph = connectLayers(lgraph, [blockId '_br1_relu2'], [concatName '/in1']);
    lgraph = connectLayers(lgraph, [blockId '_br2_relu2'], [concatName '/in2']);
    lgraph = connectLayers(lgraph, [blockId '_br3_relu2'], [concatName '/in3']);
    lgraph = connectLayers(lgraph, [blockId '_br4_relu'],  [concatName '/in4']);

    % Project back to outChannels and add residual
    proj = [
        convolution1dLayer(1, outChannels, 'Padding','same','Stride',1,'Name',[blockId '_proj_conv'])
        batchNormalizationLayer('Name',[blockId '_proj_bn'])
    ];
    addName = [blockId '_add'];
    outRelu = reluLayer('Name',[blockId '_out']);
    lgraph = addLayers(lgraph, proj);
    lgraph = addLayers(lgraph, additionLayer(2,'Name',addName));
    lgraph = addLayers(lgraph, outRelu);
    lgraph = connectLayers(lgraph, concatName, [blockId '_proj_conv']);
    lgraph = connectLayers(lgraph, [blockId '_proj_bn'], [addName '/in1']);
    lgraph = connectLayers(lgraph, inName, [addName '/in2']);
    lgraph = connectLayers(lgraph, addName, [blockId '_out']);
    outName = [blockId '_out'];
end
%% Helper Functions (RF impairments and alpha sampler)
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

