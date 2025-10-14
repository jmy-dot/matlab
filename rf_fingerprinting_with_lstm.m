%Experiment sweep controls
kValues = [1];                  % Multiplier for number of transmitters
FramesPerRouter = [300];         % Adjust for speed vs robustness
SNRList = [-5, 0, 5, 10];       % Focus on low SNR

% Define ratio of known and unknown transmitters
originalNumKnownRouters = 67;
originalNumUnknownRouters = 20;

% Resume controls
startSNR = -5;
startFramesPerRouter = FramesPerRouter(1);
startK = kValues(1);
startProcessing = true;

% Random seed for repeatability
rng(123456);%固定随机种子，保证实验可复现。实验中很多步骤存在随机性，如果不控制，每次运行代码都会产生不同结果
            %这些随机性会导致一个问题：同一套代码，今天跑的结果是85%准确率，明天跑可能变成82%，无法判断结果差异是 "算法优化导致" 还是 "随机性导致"，实验结论失去可信度。

% Speed/robustness toggles
enableParallel = false;    % Avoid parallel（并行）overhead and stalls 关闭多 worker 创建
showTrainingPlot = true;   % Show training-progress UI 显示训练进度
useLegacyImpairments = false; % true to use comm.PhaseNoise/PhaseFrequencyOffset + LUT 不使用旧版射频损伤模型
useLegacyAlpha = false;       % true to use legacy alpha sampler (mu=1.5, [1.2,2.8])  使用旧版alpha采样器（旧版均值1.5，范围[1.2,2.8]）

% Optional custom classes (Label smoothing)
%检查是否存在自定义的LabelSmoothingClassificationLayer（标签平滑分类层），如果不存在则尝试添加当前脚本路径以加载该类。
if exist('LabelSmoothingClassificationLayer','class') ~= 8
    try
        addpath(fileparts(mfilename('fullpath')));
        rehash;
    catch
    end
end

% Create main checkpoint directory with full path
% 创建主检查点目录，用于保存模型权重、训练日志等
baseCheckpointDir = fullfile(pwd, 'model_checkpoints');
if ~exist(baseCheckpointDir, 'dir')
    [mkdirSuccess, mkdirMsg] = mkdir(baseCheckpointDir);
    if ~mkdirSuccess
        warning('无法创建主检查点目录: %s。将使用当前目录保存模型。', mkdirMsg);
        baseCheckpointDir = pwd;  % 回退到当前工作目录
    end
end

%通过三层嵌套循环遍历实验参数（SNR、每路由器帧数、路由器数量倍数 k）
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
    %打印当前实验参数
    fprintf('Processing SNR = %d, FramesPerRouter = %d, k = %d\n', localSNR, localFramesPerRouter, k);

    % Problem configuration
    numKnownRouters = originalNumKnownRouters * k;
    numUnknownRouters = originalNumUnknownRouters * k;
    numTotalRouters = numKnownRouters + numUnknownRouters;

    %%这个地方
    SNR = localSNR;           % dB
    channelNumber = 1;        % WLAN channel number WLAN信道编号
    channelBand = 5;          % GHz 信道频段（5 GHz）
    frameLength = 160;        % L-LTF sequence length in samples L-LTF序列长度（160采样点，WLAN物理层前导码的一部分）
    san = 0.5;                % controls alpha distribution  控制alpha参数的分布范围（影响射频非线性程度）

    numTotalFramesPerRouter = 200;  % more data for robustness 每路由器总帧数
    %按 8:1:1 的比例划分每台路由器的信号帧，确保训练、验证、测试数据独立
    numTrainingFramesPerRouter = floor(numTotalFramesPerRouter*0.8);
    numValidationFramesPerRouter = floor(numTotalFramesPerRouter*0.1);
    numTestFramesPerRouter = numTotalFramesPerRouter - numTrainingFramesPerRouter - numValidationFramesPerRouter;

    %Per-router unique nonlinearity parameters
    %每台路由器的独特非线性参数（alpha和beta）
    %创建了两个数组存储
    all_alpha = zeros(1,numTotalRouters);
    all_beta = zeros(1,numTotalRouters);
    %选择alpha采样器（旧版/新版）
    if useLegacyAlpha
        alphaSampler = @generateAlphaLegacy;
    else
        alphaSampler = @generateAlpha;
    end
    %为每个路由器生成唯一的alpha和beta
    %模拟和控制每个路由器硬件所特有的功率放大器非线性失真
    %alpha 和 beta 被用在一个关键的数学变换公式中，直接作用于提取出的LLTF信号：
    %将原始的LLTF信号进行了一次非线性"扭曲"，生成了带有指纹特征的新LLTF。
    %在代码中为每一台模拟的路由器分配一组随机但固定的alpha, beta组合，为每台设备赋予了一个独一无二的、基于功率放大器特性的数学指纹。
    for idx = 1:numTotalRouters
        alpha = alphaSampler(san);
        beta = (alpha - 1) + 0.2 * rand(1) - 0.1;
        all_alpha(idx)= alpha;
        all_beta(idx)= beta;
    end

    % MAC/PHY configs 配置参数的定义
    %模拟真实 WLAN 信号的生成过程，包括 MAC 层信标帧（Beacon）和 PHY 层调制参数，确保生成的信号符合 802.11 协议规范，贴近实际场景。
    frameBodyConfig = wlanMACManagementConfig;
    beaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', "ManagementConfig", frameBodyConfig);
    [~, mpduLength] = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');
    %使用20 MHz的信道带宽，设置调制方式为QPSK调制
    nonHTConfig = wlanNonHTConfig('ChannelBandwidth', "CBW20", "MCS", 1, "PSDULength", mpduLength); 

    rxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
    fc = wlanChannelFrequency(channelNumber, channelBand);
    fs = wlanSampleRate(nonHTConfig);

    %对多径效应的数学模拟
    multipathChannel = comm.RayleighChannel('SampleRate', fs, ...
        'PathDelays', [0 1.8 3.4]/fs, 'AveragePathGains', [0 -2 -10], 'MaximumDopplerShift', 0);

    phaseNoiseRange = [0.01, 0.3];
    freqOffsetRange = [-4, 4];
    dcOffsetRange = [-50, -32];

    % Per-router RF impairments
    %为每个路由器生成独特的射频损伤参数
    radioImpairments = repmat(struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), numTotalRouters, 1);
    for routerIdx = 1:numTotalRouters
        radioImpairments(routerIdx).PhaseNoise = rand*(phaseNoiseRange(2)-phaseNoiseRange(1)) + phaseNoiseRange(1);
        radioImpairments(routerIdx).DCOffset = rand*(dcOffsetRange(2)-dcOffsetRange(1)) + dcOffsetRange(1);
        radioImpairments(routerIdx).FrequencyOffset = fc/1e6*(rand*(freqOffsetRange(2)-freqOffsetRange(1)) + freqOffsetRange(1));
    end

    % Pre-alloc frame buffers (complex)
    %提前为信号帧分配内存，避免后续循环中动态扩容导致的效率低下。
    xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
    xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
    xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

    %定义每类数据集的帧索引范围
    trainingIndices = 1:numTrainingFramesPerRouter;
    validationIndices = 1:numValidationFramesPerRouter;
    testIndices = 1:numTestFramesPerRouter;

    tic %启动计时器
    generatedMACAddresses = strings(numTotalRouters, 1); %初始化字符串数组，存储每个路由器的MAC地址

    % Serial frame generation单进程串行生成数据
    routerIndices = 1:numTotalRouters; %routerIndices（路由器索引）
    if useLegacyImpairments
        rfImpairFn = @helperRFImpairmentsLegacy;
    else
        rfImpairFn = @helperRFImpairments;
    end
    %从这开始，进行单进程的for循环，逐个处理每个路由器
    for idx = 1:length(routerIndices)
        routerIdx = routerIndices(idx);  %当前路由器的索引
        %生成MAC地址，已知路由器随机，未知固定
        if (routerIdx<=numKnownRouters)
            generatedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
        else
            generatedMACAddresses(idx) = 'AAAAAAAAAAAA';
        end
        % 配置Beacon帧并生成发送波形，生成 Beacon帧的数字基带波形，为后续添加信道和射频损伤做准备
        beaconFrameConfig.Address2 = generatedMACAddresses(idx); %beaconFrameConfig.Address2 设置Beacon帧的源MAC地址
        beacon = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');%wlanMACFrame：生成 MAC 层的比特流
        txWaveform = wlanWaveformGenerator(beacon, nonHTConfig); %wlanWaveformGenerator：将 MAC 层比特流调制为物理层基带波形，模拟无线网卡的数字-模拟转换过程
        txWaveform = helperNormalizeFramePower(txWaveform);%helperNormalizeFramePower：归一化信号功率，避免不同路由器的信号强度差异干扰指纹特征
        txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

        %重置多径信道
        %multipathChannel在模拟多径效应时，内部会维护一个状态变量，
        %前一个信号的传输状态会 "残留" 到下一个信号，导致不同帧/不同路由器的信号相互干扰
        %reset(multipathChannel)的作用就是清除这些残留状态，将信道重置为初始状态。
        reset(multipathChannel)
        
        %循环提取有效LLTF帧，带重试限制，避免卡死
        frameCount= 0; trials = 0; maxTrials = max(5*numTotalFramesPerRouter, 200);%重试限制
        rxLLTF = zeros(frameLength,numTotalFramesPerRouter);
        while frameCount<numTotalFramesPerRouter && trials < maxTrials  %没收集到足够的有效帧，且总尝试次数还没有达到上限。
            trials = trials + 1;

            %模拟信号的多径效应、射频损伤和添加噪声
            rxMultipath = multipathChannel(txWaveform); %在信号传输中应用多径效应
            rxImpairment = rfImpairFn(rxMultipath, radioImpairments(idx), fs); %应用射频损伤（相位噪声、频率偏移、直流偏移）
            
            %%
            %模拟无线信道中的环境噪声
            if SNR <= 0  %% 根据SNR选择噪声添加策略
                %低SNR时先加较高噪声（25dB）确保帧结构提取，后续再补回目标噪声
                %首先向原始信号rxImpairment添加一个较弱的噪声，伪造出一个临时的高信噪比环境。在这个干净的信号rxSigFE中，rxFrontEnd可以轻松地检测并提取出无损的LLTF序列。
                %使用这个语句[valid, ~, ~, ~, ~, LLTF] = rxFrontEnd(rxSigFE)提取干净的LLTF后，
                %代码会手动计算需要多大的噪声功率才能使这个LLTF的信噪比精确地达到目标低SNR值。
                rxSigFE = awgn(rxImpairment, 25, 'measured');
            else
                %正常SNR直接按目标值加噪声
                rxSigFE = awgn(rxImpairment, SNR, 'measured'); %直接使用MATLAB内置的awgn函数，将噪声添加到经过射频损伤模拟后的信号rxImpairment上，
                                                               %使其达到目标SNR值。接着，直接从这个带噪信号rxSigFE中提取LLTF。
            end
            
            %提取LLTF并判断有效性。LLTF是Wi-Fi（WLAN）信号物理层前导码的一个关键组成部分。
            [valid, ~, ~, ~, ~, LLTF] = rxFrontEnd(rxSigFE); 

            if valid  %仅保存有效帧
                %应用射频指纹非线性校正（alpha/beta）
                LLTF = LLTF.*LLTF.*all_alpha(idx) ./ (1 + all_beta(idx)* LLTF.*LLTF);
                %低SNR下补充目标噪声
                %手动计算需要多大的噪声功率才能使这个LLTF的信噪比精确地达到目标低SNR值。
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
        %处理帧数量不足的情况,用最后一帧填充或补零
        if frameCount < numTotalFramesPerRouter
            if frameCount > 0
                rxLLTF(:, frameCount+1:end) = repmat(rxLLTF(:,frameCount), 1, numTotalFramesPerRouter-frameCount);
            else
                rxLLTF(:, :) = 0;
            end
        end

        %打乱帧顺序，避免训练偏差
        %对每个路由器生成的信号帧进行随机重排，提高模型的真实泛化能力
        % 每个路由器生成的信号帧虽然来自同一设备（具有相同的射频指纹），但在生成过程中可能存在顺序相关性
        %如果直接按生成顺序将这些帧分配给训练集，模型可能会走捷径，不是学习设备的独特射频指纹，而是记住第1-10帧对应路由器A，第 11-20 帧对应路由器 B的顺序规律；
        rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));

        %将当前路由器的帧分配到训练/验证/测试集
        idxStartTrain = (idx-1)*numTrainingFramesPerRouter + 1; idxEndTrain = idx*numTrainingFramesPerRouter;
        xTrainingFrames(:, idxStartTrain:idxEndTrain) = rxLLTF(:, trainingIndices);
        idxStartVal = (idx-1)*numValidationFramesPerRouter + 1;   idxEndVal = idx*numValidationFramesPerRouter;
        xValFrames(:, idxStartVal:idxEndVal) = rxLLTF(:, validationIndices+ numTrainingFramesPerRouter);
        idxStartTest = (idx-1)*numTestFramesPerRouter + 1;        idxEndTest = idx*numTestFramesPerRouter;
        xTestFrames(:, idxStartTest:idxEndTest) = rxLLTF(:, testIndices + numTrainingFramesPerRouter+numValidationFramesPerRouter);
    end

    GenerateTime = seconds(toc); %#ok<NASGU> 单进程结束


    %%
    % Labels
    %标签生成
    labels = strings(numTotalRouters, 1);
    for idx = 1:numKnownRouters
        labels(idx) = generatedMACAddresses(idx);
    end
    labels(numKnownRouters+1:end) = "Unknown";

    %为每个帧分配对应标签
    yTrain = repelem(labels, numTrainingFramesPerRouter);
    yVal = repelem(labels, numValidationFramesPerRouter);
    yTest = repelem(labels, numTestFramesPerRouter);

    %转换为分类变量（MATLAB中分类任务的标准格式）
    yTrain = categorical(yTrain);
    yVal = categorical(yVal, categories(yTrain));
    yTest = categorical(yTest, categories(yTrain));

    %Convert complex LLTF to 2-channel features [I,Q] for LSTM
    numTrain = numel(yTrain);
    numVal = numel(yVal);
    numTest = numel(yTest);

    %原始信号是复数形式的L-LTF 序列，这里将其转换为2通道特征，适配LSTM模型
    %提取I/Q分量
    xTrain_I = real(xTrainingFrames);  xTrain_Q = imag(xTrainingFrames);
    xVal_I   = real(xValFrames);       xVal_Q   = imag(xValFrames);
    xTest_I  = real(xTestFrames);      xTest_Q  = imag(xTestFrames);
    
    %将2通道特征按列拼接（I, Q）
    xTrainingFrames = cat(2, xTrain_I, xTrain_Q);
    xValFrames      = cat(2, xVal_I,   xVal_Q);
    xTestFrames     = cat(2, xTest_I,  xTest_Q);
    
    %重塑为LSTM期望的格式：[帧长度, 通道数, 样本数]
    xTrainingFrames = reshape(xTrainingFrames, [frameLength, 2, numTrain]);
    xValFrames = reshape(xValFrames, [frameLength, 2, numVal]);
    xTestFrames = reshape(xTestFrames, [frameLength, 2, numTest]);

    % Shuffle training set
    %打乱训练集（避免模型学习顺序依赖）
    vr = randperm(numTrain);
    xTrainingFrames = xTrainingFrames(:,:,vr);
    yTrain = yTrain(vr);

    %%
    %特征归一化和数据增强部分

    %计算类别权重（解决类别不平衡问题）
    %统计每个类别的样本数量,并对每个类别的样本数取倒数（1/counts），样本越少则值越大。样本数少的类别会获得更高的权重，样本数多的类别会获得更少的权重
    %样本数少的类别（高权重）分类错误时，损失函数惩罚更大。样本数多的类别（低权重）分类错误时，惩罚较小。
    %如果模型误分类该类别：损失函数会乘以一个较大的权重值，导致总损失显著增加。
    % 通过增大少数类别的错误惩罚，迫使模型：更关注少数类别的特征。
    %代码为67个已知路由器分别创建了67个独立的类别（标签）。然后，它将33个未知路由器的所有信号帧，全部归类到了同一个、名为 "Unknown" 的类别中。
    %"Unknown" 是一个超级大类，因此classWeights 在这里的作用不是为了提升被忽略的 "Unknown"类别，
    %恰恰相反，是为了抑制巨大的 "Unknown" 类别在训练中的主导地位，强制模型去关注和学习那67个样本量极少的"已知路由器"类别的独特指纹特征。

    classNames = categories(yTrain);
    counts = countcats(yTrain);
    invCounts = 1./max(counts,1);
    classWeights = invCounts / mean(invCounts);

    %将原始特征数据转换成最适合神经网络学习的格式，并提升模型的泛化能力。
    %对训练/验证/测试集应用相同的标准化（用训练集参数，避免数据泄露）
    %全局鲁棒标准化。对所有数据进行一次全局的、基于训练集统计特性的标准化处理。

    %使用训练集计算标准化参数，标准化使不同通道特征的量纲一致（均值为 0，标准差为 1），加速模型收敛；
    allTrainData = reshape(xTrainingFrames, [], 2);  %首先，将所有训练样本压平成一个大的二维矩阵，每一行是一个时间点，每一列是一个特征通道（I, Q）。
    globalMu = median(allTrainData, 1, 'omitnan');   %计算每个特征通道的中位数。相比于均值，中位数对数据中的异常值或极端值不敏感，因此更加鲁棒。
    globalSigma = mad(allTrainData, 1, 1) * 1.4826;  %计算每个特征通道的绝对中位差，并乘以 1.4826使其成为标准差的一个鲁棒估计。同样为了避免异常值对数据规模的错误估计。
    globalSigma(globalSigma < 1e-6) = 1;             

    %使用仅从训练集中计算出的globalMu和globalSigma，对所有三个数据集进行标准化（减去中位数，再除以鲁棒标准差）。
    %为了严格遵守机器学习的准则，防止数据泄露。在真实场景中，我们永远无法提前知道未来测试数据的分布，
    %所以模型的任何预处理步骤都只能基于已有的训练数据。验证集和测试集必须被当作模拟的未来数据来处理。
    for i = 1:numTrain
        xTrainingFrames(:,:,i) = (xTrainingFrames(:,:,i) - globalMu) ./ globalSigma;
    end
    for i = 1:numVal
        xValFrames(:,:,i) = (xValFrames(:,:,i) - globalMu) ./ globalSigma;
    end
    for i = 1:numTest
        xTestFrames(:,:,i) = (xTestFrames(:,:,i) - globalMu) ./ globalSigma;
    end

    %% Build LSTM model
    inputSize = [frameLength 2 1];  %1是前一版代码为了适配图像输入层而添加的单通道占位维度，没有实际的物理或信号意义，仅用于满足深度学习框架对输入格式的要求。
                                    %体现了前一版代码在模型设计上的局限性，将时序信号强行按图像格式处理。
    numHiddenUnits = 100; 
    numClasses = numKnownRouters + 1; 

    layers = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
        % 展平层
        flattenLayer('Name', 'Flatten Input')
        % LSTM层
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
        dropoutLayer(0.5, 'Name', 'DropOut1')
        % 第二个LSTM层
        lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
        dropoutLayer(0.5, 'Name', 'DropOut2')
        % 全连接层和分类层
        fullyConnectedLayer(numClasses, 'Name', 'FC1')
        softmaxLayer('Name', 'SoftMax')
        classificationLayer('Name', 'Output')
    ];

    %% Training options
    miniBatchSize = 512; 
    iterPerEpoch = floor(numTrain/miniBatchSize);
    dirName = sprintf('snr_%d_frames_%d_k_%d', localSNR, localFramesPerRouter, k);
    dirName = strrep(dirName, ' ', '_');
    currentCheckpointDir = fullfile(baseCheckpointDir, dirName);
    if ~exist(currentCheckpointDir, 'dir')
        [mkdirSuccess, mkdirMsg] = mkdir(currentCheckpointDir);
        if ~mkdirSuccess
            warning('无法创建检查点子目录: %s。将使用主目录保存模型。', mkdirMsg);
            currentCheckpointDir = baseCheckpointDir;
        end
    end

    options = trainingOptions('adam', ...
        'MaxEpochs', 20, ...
        'ValidationData', {xValFrames, yVal}, ...
        'ValidationFrequency', iterPerEpoch, ...
        'Verbose', true, ...
        'InitialLearnRate', 0.008, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 2, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'L2Regularization', 0.001, ...
        'GradientThreshold', 1, ...
        'Plots', ternary(showTrainingPlot,'training-progress','none'), ...
        'CheckpointPath', currentCheckpointDir, ...
        'CheckpointFrequency', 1, ...
        'ExecutionEnvironment', 'auto');

    %% Train 模型训练、评估与结果保存
    tic
    fprintf('Training LSTM model...\n');
    [lastNet, trainInfo] = trainNetwork(xTrainingFrames, yTrain, layers, options); %启动深度学习模型的训练过程。
    TrainTime = seconds(toc); %#ok<NASGU>

    % Select best epoch if checkpoint exists
    [bestValAcc, bestEpochIdx] = max(trainInfo.ValidationAccuracy); %找到在验证集上准确率最高的那个时刻
    bestModelFileName = sprintf('epoch_%d_net.mat', bestEpochIdx);
    bestModelPath = fullfile(currentCheckpointDir, bestModelFileName);
    bestNet = [];
    if exist(bestModelPath,'file')
        try
            S = load(bestModelPath,'net');
            bestNet = S.net;
        catch
            bestNet = [];
        end
    end
    if isempty(bestNet)
        bestNet = lastNet;
    end

    %Evaluate
    yTestPred = classify(bestNet, xTestFrames, 'ExecutionEnvironment', 'auto'); %使用上一步选出的最佳模型，对从未见过的测试集进行预测。
    testAccuracy = mean(yTest == yTestPred); %计算并显示模型在测试集上的最终准确率。
    disp("Final model test accuracy: " + testAccuracy*100 + "%")
    figure
    cm = confusionchart(yTest, yTestPred);  %创建一个混淆矩阵图。清晰地展示模型容易将哪些类别混淆，帮助分析模型的弱点。

    %% =====================================================================
%  特征可视化分析 (t-SNE) - 最终美化版 (8个类别, 重命名, 右下角图例)
%  =====================================================================
fprintf('正在进行最终版特征可视化分析 (t-SNE)...\n');

% 步骤 1: 从训练好的网络中提取所有测试集的高维特征
featureLayer = 'LSTM2';
features_LSTM = activations(bestNet, xTestFrames, featureLayer, 'OutputAs', 'rows');

fprintf('已成功提取 %d 个样本的 %d 维特征。\n', size(features_LSTM,1), size(features_LSTM,2));

% 步骤 2: 使用t-SNE算法将特征降维至二维
rng('default'); % for reproducibility
Y_LSTM = tsne(features_LSTM, 'NumPCAComponents', 50, 'Perplexity', 30);
fprintf('t-SNE 降维完成。\n');

% 步骤 3: 选取并重命名要显示的8个类别
% 获取所有类别的名称
all_categories = categories(yTest);
% 选择前7个已知类别和"Unknown"类别
if numel(all_categories) > 8
    selected_categories_original = [all_categories(1:7); "Unknown"];
else
    selected_categories_original = all_categories; % 如果类别总数不足8，则全部显示
end

% 创建一个新的、简洁的类别名称列表
num_known_to_show = numel(selected_categories_original) - 1;
new_labels = "设备-" + string(1:num_known_to_show);
new_labels = [new_labels, "Unknown"]; % 添加Unknown标签

% 创建一个逻辑索引，只保留属于这8个类别的样本
subset_indices = ismember(yTest, categorical(selected_categories_original));

% 根据索引筛选出特征数据和对应的标签
Y_subset = Y_LSTM(subset_indices, :);
y_subset = yTest(subset_indices);

% 关键一步：重命名类别以匹配PPT风格
y_subset = renamecats(y_subset, selected_categories_original, new_labels);

fprintf('已筛选并重命名 %d 个样本用于可视化。\n', numel(y_subset));

% 步骤 4: 绘制并美化筛选后的 t-SNE 散点图
figure; % 创建一个新的图形窗口
h = gscatter(Y_subset(:,1), Y_subset(:,2), y_subset, [], 'o', 4, 'filled'); % 'o'是圆形标记, 4是标记大小, 'filled'是实心

% --- 图形美化 ---
title('LSTM 模型特征可视化 (8类别子集)', 'FontSize', 14);
xlabel('t-SNE Dimension 1', 'FontSize', 12);
ylabel('t-SNE Dimension 2', 'FontSize', 12);
grid on;
box on; % 添加边框
% !!!!!!!!!!! 以下是修改的部分：将图例放置在右下角 !!!!!!!!!!!
legend('Location', 'southeast'); 

% --- 保存图像为EMF矢量图格式 ---
tsneFileName = sprintf('LSTM_tSNE_Final_8_Classes_SNR_%d.emf', SNR);
saveas(gcf, tsneFileName, 'emf');
fprintf('最终版 t-SNE 可视化结果已保存为 EMF 矢量图: %s\n', tsneFileName);


%%
    cm.Title = 'LSTM Confusion Matrix (Test)';
    cm.RowSummary = 'row-normalized';
    confusionFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
    saveas(gcf,confusionFileName,'png');  %保存最终结果

   
    %Statistical validation (shuffles)
    numTests = 20; % keep runtime reasonable
    accuracies = zeros(numTests,1);
    for i = 1:numTests
        idx = randperm(numel(yTest));
        xTestShuffled = xTestFrames(:,:,idx);
        yTestShuffled = yTest(idx);
        yTestPred = classify(bestNet, xTestShuffled, 'ExecutionEnvironment', 'auto');
        accuracies(i) = mean(yTestShuffled == yTestPred);
    end
    averageAccuracy = mean(accuracies);
    stdAccuracy = std(accuracies);
    disp(['Average accuracy: ', num2str(averageAccuracy*100, '%.2f'), '% ± ', num2str(stdAccuracy*100, '%.2f'), '%']);

    %% Save
    saveFileName = sprintf('Optimized_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
    save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'bestNet');
    fprintf('Results saved to %s\n\n', saveFileName);
end
end
end


%% Builders and helpers
function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);
end
%射频损伤的具体实现
function y = helperRFImpairmentsLegacy(sig, radioImpairments, fs)
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);
    phaseNoise = helperGetPhaseNoise(radioImpairments);
    phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));
    impFOff = fOff(sig);
    impPhNoise = phNoise(impFOff);
    y = impPhNoise + 10^(radioImpairments.DCOffset/10);
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function y = helperNormalizeFramePower(x)
    p = mean(abs(x).^2 + eps);
    y = x ./ sqrt(p);
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
    try
        load('Mrms.mat','Mrms','MyI','xI');
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
        [~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
        phaseNoise = -abs(MyI(iRms, iFreqOffset));
    catch
        phaseNoise = -80; % fallback
    end
end

function alpha = generateAlpha(san)
    alpha = 1 + san*randn();
    alpha = min(max(alpha,0.7),1.3);
end

function alpha = generateAlphaLegacy(san)
    mu = 1.5;
    sigma = san;
    alpha = mu + sigma * randn(1, 1);
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1);
    end
end