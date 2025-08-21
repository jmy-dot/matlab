% Enhanced RF Fingerprinting with True ResNet Skip Connections + Custom Self-Attention + BiLSTM
% MATLAB R2023b compatible (no unsupported attention layers)
% Targets: robust at low SNR (-10~10 dB), >90% accuracy ambition with reasonable time

%% Experiment sweep controls
kValues = [1];                  % Multiplier for number of transmitters
FramesPerRouter = [100];        % Adjust for speed vs robustness
SNRList = [-10, -5, 0, 5, 10];  % Focus on low SNR

% Define ratio of known and unknown transmitters
originalNumKnownRouters = 67;
originalNumUnknownRouters = 33;

% Resume controls
startSNR = -10;
startFramesPerRouter = FramesPerRouter(1);
startK = kValues(1);
startProcessing = true;

% Random seed for repeatability
rng(123456);

% Speed/robustness toggles
enableParallel = false;    % Avoid parallel overhead and stalls
showTrainingPlot = true;   % Show training-progress UI
useLegacyImpairments = false; % true to use comm.PhaseNoise/PhaseFrequencyOffset + LUT
useLegacyAlpha = false;       % true to use legacy alpha sampler (mu=1.5, [1.2,2.8])

% Ensure custom layer class is accessible on path
if exist('TemporalSelfAttentionLayer','class') ~= 8
    try
        addpath(fileparts(mfilename('fullpath')));
    catch
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
    all_alpha = zeros(1,numTotalRouters);
    all_beta = zeros(1,numTotalRouters);
    % Select alpha sampler
    if useLegacyAlpha
        alphaSampler = @generateAlphaLegacy;
    else
        alphaSampler = @generateAlpha;
    end
    for idx = 1:numTotalRouters
        alpha = alphaSampler(san);
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

    % Impairment parameter ranges
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

    % Serial frame generation (robust and simple)
    routerIndices = 1:numTotalRouters;
    % Select RF impairment function
    if useLegacyImpairments
        rfImpairFn = @helperRFImpairmentsLegacy;
    else
        rfImpairFn = @helperRFImpairments;
    end
    for idx = 1:length(routerIndices)
        routerIdx = routerIndices(idx);
        if (routerIdx<=numKnownRouters)
            generatedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
        else
            generatedMACAddresses(idx) = 'AAAAAAAAAAAA';
        end

        beaconFrameConfig.Address2 = generatedMACAddresses(idx);
        beacon = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');
        txWaveform = wlanWaveformGenerator(beacon, nonHTConfig);
        txWaveform = helperNormalizeFramePower(txWaveform);
        txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

        reset(multipathChannel)

        frameCount= 0; trials = 0; maxTrials = max(5*numTotalFramesPerRouter, 200);
        rxLLTF = zeros(frameLength,numTotalFramesPerRouter);
        while frameCount<numTotalFramesPerRouter && trials < maxTrials
            trials = trials + 1;
            rxMultipath = multipathChannel(txWaveform);
            rxImpairment = rfImpairFn(rxMultipath, radioImpairments(idx), fs);

            if SNR <= 0
                rxSigFE = awgn(rxImpairment, 25, 'measured');
            else
                rxSigFE = awgn(rxImpairment, SNR, 'measured');
            end

            [valid, ~, ~, ~, ~, LLTF] = rxFrontEnd(rxSigFE);

            if valid
                % Apply PA-like nonlinearity (alpha/beta)
                LLTF = LLTF.*LLTF.*all_alpha(idx) ./ (1 + all_beta(idx)* LLTF.*LLTF);
                % Degrade to target SNR if detected at high SNR
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
        % Pad if insufficient frames
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

    GenerateTime = seconds(toc); %#ok<NASGU>

    %% Labels
    labels = generatedMACAddresses;
    labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";
    yTrain = repelem(labels, numTrainingFramesPerRouter);
    yVal = repelem(labels, numValidationFramesPerRouter);
    yTest = repelem(labels, numTestFramesPerRouter);

    %% Convert complex LLTF to 4-channel features and arrange as sequences
    % x*Frames are [frameLength x numSamples], complex
    % Channels: [I, Q, |x|, dphi]
    xTrain_I = real(xTrainingFrames);  xTrain_Q = imag(xTrainingFrames);
    xVal_I   = real(xValFrames);       xVal_Q   = imag(xValFrames);
    xTest_I  = real(xTestFrames);      xTest_Q  = imag(xTestFrames);

    xTrain_mag = sqrt(max(xTrain_I.^2 + xTrain_Q.^2, eps));
    xVal_mag   = sqrt(max(xVal_I.^2   + xVal_Q.^2,   eps));
    xTest_mag  = sqrt(max(xTest_I.^2  + xTest_Q.^2,  eps));

    xTrain_phase = atan2(xTrain_Q, xTrain_I); xTrain_dphi = [zeros(1,size(xTrain_phase,2)); diff(unwrap(xTrain_phase))];
    xVal_phase   = atan2(xVal_Q,   xVal_I);   xVal_dphi   = [zeros(1,size(xVal_phase,2));   diff(unwrap(xVal_phase))];
    xTest_phase  = atan2(xTest_Q,  xTest_I);  xTest_dphi  = [zeros(1,size(xTest_phase,2));  diff(unwrap(xTest_phase))];

    xTrainingFrames = cat(2, xTrain_I, xTrain_Q, xTrain_mag, xTrain_dphi);
    xValFrames      = cat(2, xVal_I,   xVal_Q,   xVal_mag,   xVal_dphi);
    xTestFrames     = cat(2, xTest_I,  xTest_Q,  xTest_mag,  xTest_dphi);

    % Reshape to [frameLength x 4 x N]
    numTrain = numel(yTrain);
    numVal = numel(yVal);
    numTest = numel(yTest);

    xTrainingFrames = reshape(xTrainingFrames, [frameLength, 4, numTrain]);
    xValFrames = reshape(xValFrames, [frameLength, 4, numVal]);
    xTestFrames = reshape(xTestFrames, [frameLength, 4, numTest]);

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

    % Per-sequence per-channel normalization (zero-mean, unit-variance)
    for i = 1:numTrain
        Xi = xTrainingFrames(:,:,i);
        mu = mean(Xi,1,'omitnan'); sigma = std(Xi,0,1,'omitnan'); sigma(sigma<1e-6) = 1;
        xTrainingFrames(:,:,i) = (Xi - mu) ./ sigma;
    end
    for i = 1:numVal
        Xi = xValFrames(:,:,i);
        mu = mean(Xi,1,'omitnan'); sigma = std(Xi,0,1,'omitnan'); sigma(sigma<1e-6) = 1;
        xValFrames(:,:,i) = (Xi - mu) ./ sigma;
    end
    for i = 1:numTest
        Xi = xTestFrames(:,:,i);
        mu = mean(Xi,1,'omitnan'); sigma = std(Xi,0,1,'omitnan'); sigma(sigma<1e-6) = 1;
        xTestFrames(:,:,i) = (Xi - mu) ./ sigma;
    end

    % Convert to cell arrays for sequence networks: each cell [features(=4) x time(=frameLength)]
    XTrain = cell(numTrain,1);
    for i = 1:numTrain
        Xi = squeeze(xTrainingFrames(:, :, i));     % [frameLength x 4]
        XTrain{i} = Xi.';                            % [4 x frameLength]
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

    % Stronger training-time augmentation for low SNR (training only)
    for i = 1:numTrain
        Xi = XTrain{i}; % [4 x T]
        Tlen = size(Xi,2);
        % Temporal mask
        if rand < 0.25 && Tlen > 24
            mlen = randi([8,20]);
            t0 = randi([1, max(1, Tlen-mlen+1)]);
            Xi(:, t0:min(Tlen, t0+mlen-1)) = 0;
        end
        % Random phase jitter (rotate I/Q only)
        if rand < 0.45
            theta = (pi/180) * (randn*6);
            R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            Xi(1:2,:) = R * Xi(1:2,:);
        end
        % Random gain perturbation (apply to I/Q and magnitude; do not change dphi)
        if rand < 0.35
            g = 10^(randn*0.02);
            Xi(1:3,:) = Xi(1:3,:) * g;
        end
        XTrain{i} = Xi;
    end

    %% Build True ResNet + BiLSTM model with custom attention
    inputFeatureSize = 4;           % I, Q, |x|, dphi per time step
    embedDim = 256;                 % Feature channels after CNN stack
    numClasses = numKnownRouters + 1; % include Unknown

    lgraph = layerGraph();
    [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize);

    % Optional denoise block for low SNR prior to stem
    [lgraph, lastName] = addDenoiseBlock1D(lgraph, inputName);
    % Initial 1D Conv stem
    [lgraph, lastName] = addStem1D(lgraph, lastName);

    % ResNet backbone
    [lgraph, lastName] = addResNetBackbone1D(lgraph, lastName, embedDim);

    % Extra temporal context via dilated residual blocks
    [lgraph, lastName] = addDilatedStack1D(lgraph, lastName, embedDim);

    % Channel alignment
    [lgraph, lastName] = addAlignBlock1D(lgraph, lastName, embedDim);

    % Inception-style multi-branch temporal blocks to increase connectivity
    [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc1');
    [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc2');

    % Temporal self-attention block
    [lgraph, lastName] = addSelfAttentionBlock(lgraph, lastName, embedDim);

    % BiLSTM stack
    [lgraph, lastName] = addBiLSTMStack(lgraph, lastName);

    % Classifier head
    [lgraph, lastName] = addClassifierHead(lgraph, lastName, numClasses, classNames, classWeights);

    %% Training options
    miniBatchSize = 128;
    iterPerEpoch = max(1, floor(numTrain/miniBatchSize));
    options = trainingOptions('adam', ...
        'MaxEpochs', 45, ...
        'ValidationData', {XVal, yVal}, ...
        'ValidationFrequency', max(1,ceil(iterPerEpoch/2)), ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', 3e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1, ...
        'Shuffle', 'every-epoch', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', true, ...
        'Plots', ternary(showTrainingPlot,'training-progress','none'), ...
        'OutputNetwork', 'last-iteration');

    %% Train and evaluate
    [simNet, trainInfo] = trainNetwork(XTrain, yTrain, lgraph, options); %#ok<ASGLU>

    % Report metrics
    [predVal, ~] = classify(simNet, XVal);
    valAcc = mean(predVal == yVal);
    [predTest, ~] = classify(simNet, XTest);
    testAcc = mean(predTest == yTest);

    fprintf('Best Validation Accuracy (during training): %.2f%%\n', max(trainInfo.ValidationAccuracy));
    fprintf('Final Validation Accuracy: %.2f%%\n', valAcc*100);
    fprintf('Final Test Accuracy: %.2f%%\n', testAcc*100);

end
end
end

%% Utilities and modular builders
function y = ternary(cond, a, b)
    if cond, y = a; else, y = b; end
end

function alpha = generateAlpha(san)
    alpha = 1 + san*randn();
    alpha = min(max(alpha,0.7),1.3);
end

function alpha = generateAlphaLegacy(san)
% Generate alpha with controlled variance and bounds (legacy)
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1);
    end
end

function y = helperRFImpairments(x, imp, fs)
% Apply basic RF impairments: phase noise (random walk), DC offset (dB), freq offset (ppm-based)
    N = length(x);
    % Phase noise as random-walk phase
    if imp.PhaseNoise > 0
        pnStd = imp.PhaseNoise/10; % scale
        dphi = pnStd*randn(N,1);
        phi = cumsum(dphi);
        x = x .* exp(1j*phi);
    end
    % Frequency offset (Hz)
    if isfield(imp,'FrequencyOffset') && imp.FrequencyOffset ~= 0
        n = (0:N-1).';
        x = x .* exp(1j*2*pi*imp.FrequencyOffset*n/fs);
    end
    % DC offset (dB)
    if isfield(imp,'DCOffset')
        A = 10^(imp.DCOffset/20);
        x = x + A*(1+1j);
    end
    y = x;
end

function y = helperRFImpairmentsLegacy(sig, radioImpairments, fs)
% Legacy: comm.PhaseFrequencyOffset + comm.PhaseNoise using LUT MyI/Mrms/xI
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    y = impPhNoise + 10^(radioImpairments.DCOffset/10);
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
% Legacy: get phase noise from LUT file Mrms.mat
    try
        S = load('Mrms.mat','Mrms','MyI','xI');
        Mrms = S.Mrms; MyI = S.MyI; xI = S.xI;
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        phaseNoise = -80; % fallback dBc/Hz
    end
end

%% Residual/dilated/inception/attention/LSTM/classifier builders
function [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize)
% Input layer for 1-D sequence features (e.g., 4 for I/Q/|x|/dphi)
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

function [lgraph, outName] = addResidualBlock1D(lgraph, blockName, inChannels, outChannels, stride, inputName)
% Standard residual block with projection skip if needed
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

function [lgraph, outName] = addResNetBackbone1D(lgraph, inName, embedDim)
% ResNet backbone: 64->128->embedDim with projection skips
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res1', 64, 64, 1, inName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res2', 64, 128, 2, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res3', 128, 128, 1, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res4', 128, embedDim, 2, outName);
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res5', embedDim, embedDim, 1, outName);
end

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

    % Concatenate and project
    concatName = [blockId '_concat'];
    lgraph = addLayers(lgraph, depthConcatenationLayer(4,'Name',concatName));
    lgraph = connectLayers(lgraph, [blockId '_br1_relu2'], [concatName '/in1']);
    lgraph = connectLayers(lgraph, [blockId '_br2_relu2'], [concatName '/in2']);
    lgraph = connectLayers(lgraph, [blockId '_br3_relu2'], [concatName '/in3']);
    lgraph = connectLayers(lgraph, [blockId '_br4_relu'],  [concatName '/in4']);

    proj = [
        convolution1dLayer(1, outChannels, 'Padding','same','Stride',1,'Name',[blockId '_proj1'])
        batchNormalizationLayer('Name',[blockId '_proj1_bn'])
        reluLayer('Name',[blockId '_proj1_relu'])
    ];
    lgraph = addLayers(lgraph, proj);
    lgraph = connectLayers(lgraph, concatName, [blockId '_proj1']);

    % Residual add with input (project input if needed)
    addName = [blockId '_add'];
    lgraph = addLayers(lgraph, additionLayer(2,'Name',addName));

    % Match input channels to outChannels
    match = [
        convolution1dLayer(1, outChannels, 'Padding','same','Stride',1,'Name',[blockId '_match'])
        batchNormalizationLayer('Name',[blockId '_match_bn'])
    ];
    lgraph = addLayers(lgraph, match);

    lgraph = connectLayers(lgraph, inName, [blockId '_match']);
    lgraph = connectLayers(lgraph, [blockId '_match_bn'], [addName '/in2']);
    lgraph = connectLayers(lgraph, [blockId '_proj1_relu'], [addName '/in1']);

    outRelu = reluLayer('Name',[blockId '_out']);
    lgraph = addLayers(lgraph, outRelu);
    lgraph = connectLayers(lgraph, addName, [blockId '_out']);

    outName = [blockId '_out'];
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