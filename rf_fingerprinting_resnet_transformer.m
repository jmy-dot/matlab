% Enhanced RF Fingerprinting with True ResNet Skip Connections + Transformer Self-Attention + BiLSTM (1D)
% Compatible with MATLAB R2023b
% Target: >90% accuracy, loss < 1, reasonable training time, reduce overfitting

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

            numTrainingFramesPerRouter = floor(numTotalFramesPerRouter*0.8);
            numValidationFramesPerRouter = floor(numTotalFramesPerRouter*0.1);
            numTestFramesPerRouter = numTotalFramesPerRouter - numTrainingFramesPerRouter - numValidationFramesPerRouter;

            %% Generate per-router nonlinearity parameters (alpha, beta)
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

            multipathChannel = comm.RayleighChannel(...
                'SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, ...
                'AveragePathGains', [0 -2 -10], ...
                'MaximumDopplerShift', 0);

            phaseNoiseRange = [0.01, 0.3];
            freqOffsetRange = [-4, 4];
            dcOffsetRange = [-50, -32];

            rng(123456)

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

            % Pre-allocate complex frames (frameLength x totalFrames)
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

                        % Router-specific nonlinearity
                        LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                        if valid
                            frameCount=frameCount+1;
                            rxLLTF(:,frameCount) = LLTF;
                        end
                    end

                    % Shuffle frames per router
                    rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));

                    % Split into train/val/test
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
            GenerateTime = seconds(toc);
            toc

            %% Labels
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter).';
            yVal = repelem(labels, numValidationFramesPerRouter).';
            yTest = repelem(labels, numTestFramesPerRouter).';

            %% Convert complex frames to sequence cell arrays (features x time)
            % Each sample: 2 x frameLength (I,Q as features), time dimension = frameLength
            numTrain = size(xTrainingFrames,2);
            numVal = size(xValFrames,2);
            numTest = size(xTestFrames,2);

            xTrainSeq = cell(numTrain,1);
            for n = 1:numTrain
                seq = [real(xTrainingFrames(:,n)).'; imag(xTrainingFrames(:,n)).']; % 2 x T
                xTrainSeq{n} = seq;
            end
            xValSeq = cell(numVal,1);
            for n = 1:numVal
                seq = [real(xValFrames(:,n)).'; imag(xValFrames(:,n)).'];
                xValSeq{n} = seq;
            end
            xTestSeq = cell(numTest,1);
            for n = 1:numTest
                seq = [real(xTestFrames(:,n)).'; imag(xTestFrames(:,n)).'];
                xTestSeq{n} = seq;
            end

            % Shuffle training set
            vr = randperm(numTrain);
            xTrainSeq = xTrainSeq(vr);
            yTrain = categorical(yTrain(vr));

            yVal = categorical(yVal);
            yTest = categorical(yTest);

            %% Build Hybrid 1D ResNet + Transformer Self-Attention + BiLSTM
            numClasses = numKnownRouters + 1; % known + Unknown

            embedDim = 256;    % model channel dim before transformer
            attnHeads = 4;     % number of attention heads
            ffDim = 512;       % transformer feed-forward dimension

            lgraph = layerGraph();

            % Input
            lgraph = addLayers(lgraph, sequenceInputLayer(2, 'Name', 'seq_input', 'Normalization', 'none'));

            % Stem: expand to 64 channels
            stem = [ ...
                convolution1dLayer(7, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'stem_conv');
                layerNormalizationLayer('Name', 'stem_ln');
                reluLayer('Name', 'stem_relu')
            ];
            lgraph = addLayers(lgraph, stem);
            lgraph = connectLayers(lgraph, 'seq_input', 'stem_conv');

            % ResNet Stage 1: 64 channels, 2 blocks, no downsample first block
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's1_b1', 64, 3, 1, 'stem_relu');
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's1_b2', 64, 3, 1, lastName);

            % Downsample to Stage 2, increase channels to 128
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's2_b1', 128, 3, 2, lastName); % downsample
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's2_b2', 128, 3, 1, lastName);

            % Downsample to Stage 3, increase channels to 256
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's3_b1', 256, 3, 2, lastName);
            [lgraph, lastName] = addResidualBlock1d(lgraph, 's3_b2', 256, 3, 1, lastName);

            % Normalize before attention
            lgraph = addLayers(lgraph, layerNormalizationLayer('Name', 'pre_attn_ln'));
            lgraph = connectLayers(lgraph, lastName, 'pre_attn_ln');

            % Project channels to embedDim for transformer
            proj = convolution1dLayer(1, embedDim, 'Padding', 'same', 'Name', 'proj_to_embed');
            lgraph = addLayers(lgraph, proj);
            lgraph = connectLayers(lgraph, 'pre_attn_ln', 'proj_to_embed');

            % Transformer encoder for self-attention over time
            enc = transformerEncoderLayer(attnHeads, embedDim, ffDim, ...
                'Name', 'encoder', 'Dropout', 0.1);
            lgraph = addLayers(lgraph, enc);
            lgraph = connectLayers(lgraph, 'proj_to_embed', 'encoder');

            % BiLSTM for temporal summarization (last)
            bilstm = [ ...
                bilstmLayer(128, 'OutputMode', 'last', 'Name', 'bilstm');
                dropoutLayer(0.3, 'Name', 'bilstm_dropout')
            ];
            lgraph = addLayers(lgraph, bilstm);
            lgraph = connectLayers(lgraph, 'encoder', 'bilstm');

            % Classifier
            classifier = [ ...
                fullyConnectedLayer(512, 'Name', 'fc1');
                reluLayer('Name', 'fc1_relu');
                dropoutLayer(0.5, 'Name', 'fc1_dropout');
                fullyConnectedLayer(numClasses, 'Name', 'fc_out');
                softmaxLayer('Name', 'softmax');
                classificationLayer('Name', 'cls')
            ];
            lgraph = addLayers(lgraph, classifier);
            lgraph = connectLayers(lgraph, 'bilstm_dropout', 'fc1');

            %% Training options
            miniBatchSize = 128;
            iterPerEpoch = floor(numTrain/miniBatchSize);
            if iterPerEpoch < 1
                iterPerEpoch = 1;
            end

            options = trainingOptions('adam', ...
                'MaxEpochs', 30, ...
                'ValidationData', {xValSeq, yVal}, ...
                'ValidationFrequency', max(1, iterPerEpoch), ...
                'Verbose', false, ...
                'InitialLearnRate', 1e-3, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.5, ...
                'LearnRateDropPeriod', 10, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 1e-4, ...
                'GradientThreshold', 1, ...
                'ExecutionEnvironment', 'auto');

            tic
            fprintf('Training ResNet+Transformer+BiLSTM model...\n');
            simNet = trainNetwork(xTrainSeq, yTrain, lgraph, options);
            TrainTime = seconds(toc);
            disp("Model training time = ");
            toc

            %% Test the model
            yTestPred = classify(simNet, xTestSeq, 'ExecutionEnvironment', 'auto');

            testAccuracy = mean(yTest == yTestPred);
            disp("Model test accuracy: " + testAccuracy*100 + "%")
            figure
            cm = confusionchart(yTest, yTestPred);
            cm.Title = 'ResNet+Transformer+BiLSTM Confusion Matrix for Test Data';
            cm.RowSummary = 'row-normalized';
            confusionFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
            saveas(gcf,confusionFileName,'png');

            %% Multiple test runs for statistical validation (shuffling)
            numTests = 50; % tuned for runtime
            accuracies = zeros(numTests,1);
            losses = zeros(numTests,1);

            for i = 1:numTests
                idx = randperm(numel(yTest));
                xTestSeqShuffled = xTestSeq(idx);
                yTestShuffled = yTest(idx);

                [yTestPred, scores] = classify(simNet, xTestSeqShuffled, 'ExecutionEnvironment', 'auto');
                accuracies(i) = mean(yTestShuffled == yTestPred);

                % Compute cross-entropy loss for the batch
                lossSum = 0;
                for j = 1:numel(yTestShuffled)
                    c = double(yTestShuffled(j));
                    lossSum = lossSum - log(max(scores(j,c), 1e-8));
                end
                losses(i) = lossSum/numel(yTestShuffled);
            end

            averageAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            averageLoss = mean(losses);

            disp(['Average accuracy: ', num2str(averageAccuracy*100), '% ± ', num2str(stdAccuracy*100), '%']);
            disp(['Average loss: ', num2str(averageLoss)]);

            %% Save Results
            saveFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
            save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'averageLoss', 'simNet');
            fprintf('Results saved to %s\n\n', saveFileName);
        end
    end
end

%% Helper: Add a 1D residual block with optional downsampling
function [lgraph, lastName] = addResidualBlock1d(lgraph, namePrefix, numFilters, filterSize, stride, inputName)
% Main path
conv1 = convolution1dLayer(filterSize, numFilters, 'Padding', 'same', 'Stride', stride, 'Name', [namePrefix '_conv1']);
ln1 = layerNormalizationLayer('Name', [namePrefix '_ln1']);
relu1 = reluLayer('Name', [namePrefix '_relu1']);
conv2 = convolution1dLayer(filterSize, numFilters, 'Padding', 'same', 'Stride', 1, 'Name', [namePrefix '_conv2']);
ln2 = layerNormalizationLayer('Name', [namePrefix '_ln2']);
add = additionLayer(2, 'Name', [namePrefix '_add']);
relu2 = reluLayer('Name', [namePrefix '_out']);

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, ln1);
lgraph = addLayers(lgraph, relu1);
lgraph = addLayers(lgraph, conv2);
lgraph = addLayers(lgraph, ln2);
lgraph = addLayers(lgraph, add);
lgraph = addLayers(lgraph, relu2);

% Shortcut path (projection if needed)
if stride ~= 1
    proj = convolution1dLayer(1, numFilters, 'Padding', 'same', 'Stride', stride, 'Name', [namePrefix '_proj']);
    lgraph = addLayers(lgraph, proj);
    lgraph = connectLayers(lgraph, inputName, [namePrefix '_proj']);
    shortcutName = [namePrefix '_proj'];
else
    shortcutName = inputName;
end

% Connect main path
lgraph = connectLayers(lgraph, inputName, [namePrefix '_conv1']);
lgraph = connectLayers(lgraph, [namePrefix '_conv1'], [namePrefix '_ln1']);
lgraph = connectLayers(lgraph, [namePrefix '_ln1'], [namePrefix '_relu1']);
lgraph = connectLayers(lgraph, [namePrefix '_relu1'], [namePrefix '_conv2']);
lgraph = connectLayers(lgraph, [namePrefix '_conv2'], [namePrefix '_ln2']);

% Add and out
lgraph = connectLayers(lgraph, [namePrefix '_ln2'], [namePrefix '_add/in1']);
lgraph = connectLayers(lgraph, shortcutName, [namePrefix '_add/in2']);
lgraph = connectLayers(lgraph, [namePrefix '_add'], [namePrefix '_out']);

lastName = [namePrefix '_out'];
end

%% Helper Functions

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
% helperRFImpairments Apply RF impairments
%   IMPAIREDSIG = helperRFImpairments(SIG, RADIOIMPAIRMENTS, FS) returns signal
%   SIG after applying the impairments defined by RADIOIMPAIRMENTS
%   structure at the sample rate FS.

% Apply frequency offset
fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);

% Apply phase noise
phaseNoise = helperGetPhaseNoise(radioImpairments);
phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));

impFOff = fOff(sig);
impPhNoise = phNoise(impFOff);

% Apply DC offset
impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);

end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
% helperGetPhaseNoise Get phase noise value
load('Mrms.mat','Mrms','MyI','xI');
[~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
[~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
phaseNoise = -abs(MyI(iRms, iFreqOffset));
end

function alpha = generateAlpha(san)
% Function to generate alpha with controlled variance
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);

    % Ensure alpha is within the specified range
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1);
    end
end

function y = helperNormalizeFramePower(x)
% Normalize average power to 1
    p = mean(abs(x).^2 + eps);
    y = x ./ sqrt(p);
end