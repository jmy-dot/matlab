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

% Model selection: 'MSRFNet' or 'LSTM' or 'Both'
modelType = 'Both';  % 选择要训练的模型类型

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

    %Convert complex LLTF to 4-channel features [I,Q,|x|,dphi]
    numTrain = numel(yTrain);
    numVal = numel(yVal);
    numTest = numel(yTest);

    %原始信号是复数形式的L-LTF 序列，这里将其转换为4通道特征，更全面地捕捉射频指纹
    %提取I/Q分量
    xTrain_I = real(xTrainingFrames);  xTrain_Q = imag(xTrainingFrames);
    xVal_I   = real(xValFrames);       xVal_Q   = imag(xValFrames);
    xTest_I  = real(xTestFrames);      xTest_Q  = imag(xTestFrames);
    %提取幅度特征
    xTrain_mag = sqrt(max(xTrain_I.^2 + xTrain_Q.^2, eps));
    xVal_mag   = sqrt(max(xVal_I.^2   + xVal_Q.^2,   eps));
    xTest_mag  = sqrt(max(xTest_I.^2  + xTest_Q.^2,  eps));
    %提取相位差特征
    %相位差直接关联于瞬时频率偏移，而频率偏移是射频振荡器稳定性的一个关键指纹特征。
    xTrain_phase = atan2(xTrain_Q, xTrain_I); %使用atan2函数计算出每个采样点的瞬时相位，并将其存储在名为 xTrain_phase 的变量中。
    xTrain_dphi = [zeros(1,size(xTrain_phase,2)); 
    diff(unwrap(xTrain_phase))]; %使用 diff(unwrap(...)) 对瞬时相位进行处理，计算出相邻采样点之间的相位变化量，也就是相位差，并将其存储在 xTrain_dphi 变量中。
    xVal_phase   = atan2(xVal_Q,   xVal_I);   xVal_dphi   = [zeros(1,size(xVal_phase,2));   diff(unwrap(xVal_phase))];
    xTest_phase  = atan2(xTest_Q,  xTest_I);  xTest_dphi  = [zeros(1,size(xTest_phase,2));  diff(unwrap(xTest_phase))];
    %将4通道特征按列拼接（I, Q, 幅度, 相位差）
    xTrainingFrames = cat(2, xTrain_I, xTrain_Q, xTrain_mag, xTrain_dphi);
    xValFrames      = cat(2, xVal_I,   xVal_Q,   xVal_mag,   xVal_dphi);
    xTestFrames     = cat(2, xTest_I,  xTest_Q,  xTest_mag,  xTest_dphi);
    %重塑为3D张量：[帧长度, 通道数, 样本数]
    xTrainingFrames = reshape(xTrainingFrames, [frameLength, 4, numTrain]);
    xValFrames = reshape(xValFrames, [frameLength, 4, numVal]);
    xTestFrames = reshape(xTestFrames, [frameLength, 4, numTest]);

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
    allTrainData = reshape(xTrainingFrames, [], 4);  %首先，将所有训练样本压平成一个大的二维矩阵，每一行是一个时间点，每一列是一个特征通道（I, Q, 幅度, 相位差）。
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

    %对每个独立的信号序列进行精细化处理，并为训练集增加多样性。

    %转换为元胞数组，并进行序列级处理。神经网络的序列输入层通常接受元胞数组作为输入，其中每个元胞包含一个独立的样本序列。
    XTrain = cell(numTrain,1);
    %XTrain进行序列归一化、数据增强和再次归一化
    for i = 1:numTrain
        Xi = squeeze(xTrainingFrames(:, :, i)).';  %转置 将数据从 [帧长 x 通道数] 变成 [通道数 x 帧长]，这是序列层期望的格式。
        Xi = normalizePerSequence(Xi);             %序列级归一化 对当前这一个信号序列进行归一化，消除不同帧之间整体信号强度的差异，让模型更专注于信号内部的形态特征。
        Xi = augmentSequence(Xi, SNR);             %数据增强 只对训练集进行随机的变换，如添加少量噪声、随机旋转相位、随机掩盖一小段数据等。
                                                   %极大地增加了训练数据的多样性，是一种非常有效的正则化手段，可以防止模型过拟合，提升其在未知数据上的表现。
        Xi = normalizePerSequence(Xi);             %再次序列级归一化 确保输入到网络中的数据是干净的。
        XTrain{i} = Xi;
    end
    %XVal和XTest只进行序列归一化，没有数据增强
    %数据增强（如随机掩码、相位旋转等）是一种正则化手段，目的是提高模型的泛化能力，仅适用于训练集。
    %验证集和测试集需要保持原始分布，才能真实反映模型的泛化性能，因此不进行增强。
    XVal = cell(numVal,1);
    for i = 1:numVal
        Xi = squeeze(xValFrames(:, :, i)).';
        Xi = normalizePerSequence(Xi);
        XVal{i} = Xi;
    end
    XTest = cell(numTest,1);
    for i = 1:numTest
        Xi = squeeze(xTestFrames(:, :, i)).';
        Xi = normalizePerSequence(Xi);
        XTest{i} = Xi;
    end

    %% 为LSTM模型准备数据格式
    % 将复数数据转换为实数和虚数部分
    xTrain_LSTM = [real(xTrainingFrames(:)), imag(xTrainingFrames(:))];
    xVal_LSTM = [real(xValFrames(:)), imag(xValFrames(:))];
    xTest_LSTM = [real(xTestFrames(:)), imag(xTestFrames(:))];
    
    % 重塑为LSTM期望的格式 [frameLength, 2, numSamples]
    xTrain_LSTM = permute(reshape(xTrain_LSTM, [frameLength, numTrain, 2]), [1 3 2]);
    xVal_LSTM = permute(reshape(xVal_LSTM, [frameLength, numVal, 2]), [1 3 2]);
    xTest_LSTM = permute(reshape(xTest_LSTM, [frameLength, numTest, 2]), [1 3 2]);
    
    % 打乱LSTM训练数据
    vr_lstm = randperm(numTrain);
    xTrain_LSTM = xTrain_LSTM(:,:,vr_lstm);
    yTrain_LSTM = yTrain(vr_lstm);

    %% 训练MSRFNet模型
    if strcmp(modelType, 'MSRFNet') || strcmp(modelType, 'Both')
        fprintf('Training MSRFNet model...\n');
        
        %% Build ResNet + Inception + TCN + BiLSTM model
        inputFeatureSize = 4; %输入特征通道数，I/Q/幅度/相位差
        embedDim = 320; %中间特征嵌入维度
        numClasses = numel(classNames); %分类类别数
        lgraph = layerGraph(); 
        % 输入层 定义输入格式
        [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize); %输入维度：[4, 160]（序列长度 × 特征通道数，单样本）输出维度不变
        %%
        %阶段一，信号预处理与初步特征提取
        %去噪残差块
        [lgraph, lastName] = addDenoiseBlock1D(lgraph, inputName);%减轻低 SNR下的噪声干扰，通过残差连接保留原始特征。输出维度：[8, 160]
        %主干层
        [lgraph, lastName] = addStem1D(lgraph, lastName);%初步提取高阶特征，压缩空间维度，输出维度：[64, 160]
        %%
        %阶段二，深度时序特征建模
        %ResNet主干网络
        [lgraph, lastName] = addResNetBackbone1D(lgraph, lastName, embedDim);%通过残差连接构建深层网络，提取多尺度特征，逐步扩大通道数。
                                                                             %输出维度：[320, 40],序列长度压缩4倍，通道数扩展到320
        %空洞残差块                                                                    
        [lgraph, lastName] = addDilatedStack1D(lgraph, lastName, embedDim);%通过空洞卷积扩大感受野，捕捉长距离时序依赖。输出维度：[320, 40]不变
        %对齐块
        [lgraph, lastName] = addAlignBlock1D(lgraph, lastName, embedDim);%1×1卷积调整特征通道的线性组合，BN层标准化特征分布，确保后续模块输入稳定。输出维度[320, 40]不变
        %Inception 模块
        [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc1');%多分支并行提取不同尺度的时序特征，增强特征多样性。输出维度：[320, 40]不变
        [lgraph, lastName] = addInceptionDilated1D(lgraph, lastName, embedDim, 'inc2');
        %TCN模块
        [lgraph, lastName] = addTCNBlock1D(lgraph, lastName, embedDim);%时序卷积网络（TCN）进一步捕捉长时序依赖，增强序列建模能力。输出维度不变
        %%
        %阶段三，序列聚合与分类 
        %BiLSTM 
        [lgraph, lastName] = addBiLSTMStackStrong(lgraph, lastName);%建模序列的双向时序依赖，捕捉前后文关联的指纹特征。输出维度：[384, 1]（最后一层LSTM的输出向量）
        %分类头
        [lgraph, lastName] = addClassifierHeadLS(lgraph, lastName, numClasses, classNames, classWeights);%将LSTM输出的特征向量映射到分类标签，输出每个类别的概率。输出维度：[numClasses, 1]
        %% Training options
        miniBatchSize = 96; %小批量的样本数
        iterPerEpoch = max(1, floor(numTrain/miniBatchSize));%每个epoch的迭代次数，由"总训练样本数÷批量大小（96）" 计算得到。例如若有960个训练样本，每个 epoch 将迭代10次。
        dirName = sprintf('MSRFNet_snr_%d_frames_%d_k_%d', localSNR, localFramesPerRouter, k);
        dirName = strrep(dirName, ' ', '_');
        currentCheckpointDir = fullfile(baseCheckpointDir, dirName);
        if ~exist(currentCheckpointDir, 'dir')
            [mkdirSuccess, mkdirMsg] = mkdir(currentCheckpointDir);
            if ~mkdirSuccess
                warning('无法创建检查点子目录: %s。将使用主目录保存模型。', mkdirMsg);
                currentCheckpointDir = baseCheckpointDir;
            end
        end

        options = trainingOptions('adam', ...%使用 Adam 优化器 自适应学习率，为模型中的每一个参数都独立计算一个学习率。对于那些梯度变化平缓的参数，它会用较大的步长；而对于梯度变化剧烈的参数，它会谨慎地使用较小的步长。这种自适应调整的能力使其在各种不同的模型和数据上都表现稳健。
            'MaxEpochs', 30, ...最大训练周期为30轮
            'ValidationData', {XVal, yVal}, ...监控模型在非训练数据上的性能，避免过拟合。
            'ValidationFrequency', max(1,ceil(iterPerEpoch/2)), ...%验证频率设置为每半轮验证 1 次
            'Verbose', true, ...在命令行窗口打印实时进度和关键信息。
            'InitialLearnRate', 8e-4, ...初始学习率为 0.0008，控制参数更新的步长，学习率决定了每次参数更新的幅度，参数=参数-学习率×梯度
            'LearnRateSchedule', 'piecewise', ...学习率调度策略为 "分段衰减"，按固定周期降低学习率。
            'LearnRateDropFactor', 0.6, ...学习率衰减因子为 0.6，即每次衰减时学习率变为原来的 60%
            'LearnRateDropPeriod', 8, ...每训练 8 轮衰减一次学习率。
            'MiniBatchSize', miniBatchSize, ...每次参数更新使用的样本数
            'Shuffle', 'every-epoch', ...每轮训练前打乱训练数据的顺序
            'L2Regularization', 1e-4, ...L2正则化系数为 0.0001，抑制参数过大，避免模型过度拟合训练集中的噪声
            'GradientThreshold', 1, ...梯度裁剪阈值为1，当梯度的L2范数超过1 时，按比例缩放梯度使其不超过阈值。解决训练过程中的 "梯度爆炸" 问题
            'Plots', ternary(showTrainingPlot,'training-progress','none'), ...显示训练进度图表
            'CheckpointPath', currentCheckpointDir, ...
            'CheckpointFrequency', 1, ...每训练 1 轮（epoch）保存一次检查点
            'ExecutionEnvironment', 'auto');%自动选择训练环境

        %% Train MSRFNet模型训练、评估与结果保存
        tic
        fprintf('Training ResNet + Inception + TCN + BiLSTM model...\n');
        [lastNet_MSRF, trainInfo_MSRF] = trainNetwork(XTrain, yTrain, lgraph, options); %启动深度学习模型的训练过程。
        TrainTime_MSRF = seconds(toc); %#ok<NASGU>

        % Select best epoch if checkpoint exists
        [bestValAcc_MSRF, bestEpochIdx_MSRF] = max(trainInfo_MSRF.ValidationAccuracy); %找到在验证集上准确率最高的那个时刻
        bestModelFileName = sprintf('epoch_%d_net.mat', bestEpochIdx_MSRF);
        bestModelPath = fullfile(currentCheckpointDir, bestModelFileName);
        bestNet_MSRF = [];
        if exist(bestModelPath,'file')
            try
                S = load(bestModelPath,'net');
                bestNet_MSRF = S.net;
            catch
                bestNet_MSRF = [];
            end
        end
        if isempty(bestNet_MSRF)
            bestNet_MSRF = lastNet_MSRF;
        end

        %Evaluate MSRFNet
        yTestPred_MSRF = classify(bestNet_MSRF, XTest, 'ExecutionEnvironment', 'auto'); %使用上一步选出的最佳模型，对从未见过的测试集进行预测。
        testAccuracy_MSRF = mean(yTest == yTestPred_MSRF); %计算并显示模型在测试集上的最终准确率。
        disp("MSRFNet Final model test accuracy: " + testAccuracy_MSRF*100 + "%")
        
        % 保存MSRFNet混淆矩阵
        figure
        cm_MSRF = confusionchart(yTest, yTestPred_MSRF);  %创建一个混淆矩阵图。清晰地展示模型容易将哪些类别混淆，帮助分析模型的弱点。
        cm_MSRF.Title = 'MSRFNet Confusion Matrix (Test)';
        cm_MSRF.RowSummary = 'row-normalized';
        confusionFileName_MSRF = sprintf('MSRFNet_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
        saveas(gcf,confusionFileName_MSRF,'png');  %保存最终结果
    end

    %% 训练LSTM模型
    if strcmp(modelType, 'LSTM') || strcmp(modelType, 'Both')
        fprintf('Training LSTM model...\n');
        
        % LSTM模型定义
        inputSize = [frameLength 2 1];  %1是前一版代码为了适配图像输入层而添加的单通道占位维度，没有实际的物理或信号意义，仅用于满足深度学习框架对输入格式的要求。
                                        %体现了前一版代码在模型设计上的局限性，将时序信号强行按图像格式处理。
        numHiddenUnits = 100; 
        numClasses = numKnownRouters + 1; 

        layers_LSTM = [
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

        miniBatchSize_LSTM = 512; 
        iterPerEpoch_LSTM = floor(numTrain/miniBatchSize_LSTM);

        options_LSTM = trainingOptions('adam', ...
            'MaxEpochs', 20, ...
            'ValidationData', {xVal_LSTM, yVal}, ...
            'ValidationFrequency', iterPerEpoch_LSTM, ...
            'Verbose', false, ...
            'InitialLearnRate', 0.008, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 2, ...
            'MiniBatchSize', miniBatchSize_LSTM, ...
            'Plots', ternary(showTrainingPlot,'training-progress','none'), ...
            'Shuffle', 'every-epoch', ...
            'L2Regularization', 0.001,...
            'ExecutionEnvironment', 'cpu');  
        
        tic
        simNet_LSTM = trainNetwork(xTrain_LSTM, yTrain_LSTM, layers_LSTM, options_LSTM);
        TrainTime_LSTM = seconds(toc);

        disp("LSTM training time = ");
        toc

        % Evaluate LSTM
        yTestPred_LSTM = classify(simNet_LSTM, xTest_LSTM, 'ExecutionEnvironment', 'cpu');
        testAccuracy_LSTM = mean(yTest == yTestPred_LSTM);
        disp("LSTM Test accuracy: " + testAccuracy_LSTM*100 + "%")
        
        % 保存LSTM混淆矩阵
        figure
        cm_LSTM = confusionchart(yTest, yTestPred_LSTM);
        cm_LSTM.Title = 'LSTM Confusion Matrix (Test)';
        cm_LSTM.RowSummary = 'row-normalized';
        confusionFileName_LSTM = sprintf('LSTM_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
        saveas(gcf,confusionFileName_LSTM,'png');
    end

    %% t-SNE特征可视化对比
    if strcmp(modelType, 'Both')
        fprintf('正在进行t-SNE特征可视化对比分析...\n');
        
        % 提取MSRFNet特征
        featureLayer_MSRF = 'rnn_drop2';
        features_MSRFNet = activations(bestNet_MSRF, XTest, featureLayer_MSRF, 'OutputAs', 'rows');
        fprintf('已成功提取MSRFNet %d 个样本的 %d 维特征。\n', size(features_MSRFNet,1), size(features_MSRFNet,2));

        % 提取LSTM特征 (从最后一个LSTM层)
        featureLayer_LSTM = 'LSTM2';
        features_LSTM = activations(simNet_LSTM, xTest_LSTM, featureLayer_LSTM, 'OutputAs', 'rows');
        fprintf('已成功提取LSTM %d 个样本的 %d 维特征。\n', size(features_LSTM,1), size(features_LSTM,2));

        % 使用t-SNE降维
        rng('default'); % for reproducibility
        Y_MSRFNet = tsne(features_MSRFNet, 'NumPCAComponents', 50, 'Perplexity', 30);
        Y_LSTM = tsne(features_LSTM, 'NumPCAComponents', 50, 'Perplexity', 30);
        fprintf('t-SNE 降维完成。\n');

        % 选取并重命名要显示的8个类别
        all_categories = categories(yTest);
        if numel(all_categories) > 8
            selected_categories_original = [all_categories(1:7); "Unknown"];
        else
            selected_categories_original = all_categories;
        end

        % 创建新的、简洁的类别名称列表
        num_known_to_show = numel(selected_categories_original) - 1;
        new_labels = "设备-" + string(1:num_known_to_show);
        new_labels = [new_labels, "Unknown"];

        % 创建逻辑索引，只保留属于这8个类别的样本
        subset_indices = ismember(yTest, categorical(selected_categories_original));
        Y_MSRFNet_subset = Y_MSRFNet(subset_indices, :);
        Y_LSTM_subset = Y_LSTM(subset_indices, :);
        y_subset = yTest(subset_indices);

        % 重命名类别以匹配PPT风格
        y_subset = renamecats(y_subset, selected_categories_original, new_labels);

        fprintf('已筛选并重命名 %d 个样本用于可视化。\n', numel(y_subset));

        % 绘制对比图
        figure('Position', [100, 100, 1200, 500]);
        
        % MSRFNet t-SNE图
        subplot(1,2,1);
        h1 = gscatter(Y_MSRFNet_subset(:,1), Y_MSRFNet_subset(:,2), y_subset, [], 'o', 4, 'filled');
        title('MSRFNet 模型特征可视化', 'FontSize', 14);
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        grid on; box on;
        legend('Location', 'southeast');

        % LSTM t-SNE图
        subplot(1,2,2);
        h2 = gscatter(Y_LSTM_subset(:,1), Y_LSTM_subset(:,2), y_subset, [], 'o', 4, 'filled');
        title('LSTM 模型特征可视化', 'FontSize', 14);
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        grid on; box on;
        legend('Location', 'southeast');

        % 保存对比图
        tsneFileName = sprintf('Model_Comparison_tSNE_SNR_%d_Frame_%d.emf', SNR, localFramesPerRouter);
        saveas(gcf, tsneFileName, 'emf');
        fprintf('模型对比 t-SNE 可视化结果已保存为 EMF 矢量图: %s\n', tsneFileName);

        % 分别保存两个模型的t-SNE图
        figure;
        h = gscatter(Y_MSRFNet_subset(:,1), Y_MSRFNet_subset(:,2), y_subset, [], 'o', 4, 'filled');
        title('MSRFNet 模型特征可视化 (8类别子集)', 'FontSize', 14);
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        grid on; box on;
        legend('Location', 'southeast');
        tsneFileName_MSRF = sprintf('MSRFNet_tSNE_Final_8_Classes_SNR_%d.emf', SNR);
        saveas(gcf, tsneFileName_MSRF, 'emf');

        figure;
        h = gscatter(Y_LSTM_subset(:,1), Y_LSTM_subset(:,2), y_subset, [], 'o', 4, 'filled');
        title('LSTM 模型特征可视化 (8类别子集)', 'FontSize', 14);
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        grid on; box on;
        legend('Location', 'southeast');
        tsneFileName_LSTM = sprintf('LSTM_tSNE_Final_8_Classes_SNR_%d.emf', SNR);
        saveas(gcf, tsneFileName_LSTM, 'emf');
    end

    %% 统计验证
    if strcmp(modelType, 'MSRFNet') || strcmp(modelType, 'Both')
        numTests = 20; % keep runtime reasonable
        accuracies_MSRF = zeros(numTests,1);
        for i = 1:numTests
            idx = randperm(numel(yTest));
            XTestShuffled = XTest(idx);
            yTestShuffled = yTest(idx);
            yTestPred = classify(bestNet_MSRF, XTestShuffled, 'ExecutionEnvironment', 'auto');
            accuracies_MSRF(i) = mean(yTestShuffled == yTestPred);
        end
        averageAccuracy_MSRF = mean(accuracies_MSRF);
        stdAccuracy_MSRF = std(accuracies_MSRF);
        disp(['MSRFNet Average accuracy: ', num2str(averageAccuracy_MSRF*100, '%.2f'), '% ± ', num2str(stdAccuracy_MSRF*100, '%.2f'), '%']);
    end

    if strcmp(modelType, 'LSTM') || strcmp(modelType, 'Both')
        numTests = 20;
        accuracies_LSTM = zeros(numTests,1);
        for i = 1:numTests
            idx = randperm(numel(yTest));
            xTestShuffled = xTest_LSTM(:,:,idx);
            yTestShuffled = yTest(idx);
            yTestPred = classify(simNet_LSTM, xTestShuffled, 'ExecutionEnvironment', 'cpu');
            accuracies_LSTM(i) = mean(yTestShuffled == yTestPred);
        end
        averageAccuracy_LSTM = mean(accuracies_LSTM);
        stdAccuracy_LSTM = std(accuracies_LSTM);
        disp(['LSTM Average accuracy: ', num2str(averageAccuracy_LSTM*100, '%.2f'), '% ± ', num2str(stdAccuracy_LSTM*100, '%.2f'), '%']);
    end

    %% 保存结果
    if strcmp(modelType, 'Both')
        saveFileName = sprintf('Comparison_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters, SNR, localFramesPerRouter);
        save(saveFileName, 'GenerateTime', 'TrainTime_MSRF', 'TrainTime_LSTM', ...
             'testAccuracy_MSRF', 'testAccuracy_LSTM', 'averageAccuracy_MSRF', 'stdAccuracy_MSRF', ...
             'averageAccuracy_LSTM', 'stdAccuracy_LSTM', 'bestNet_MSRF', 'simNet_LSTM');
        fprintf('Comparison results saved to %s\n\n', saveFileName);
    elseif strcmp(modelType, 'MSRFNet')
        saveFileName = sprintf('MSRFNet_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters, SNR, localFramesPerRouter);
        save(saveFileName, 'GenerateTime', 'TrainTime_MSRF', 'averageAccuracy_MSRF', 'stdAccuracy_MSRF', 'bestNet_MSRF');
        fprintf('MSRFNet results saved to %s\n\n', saveFileName);
    elseif strcmp(modelType, 'LSTM')
        saveFileName = sprintf('LSTM_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters, SNR, localFramesPerRouter);
        save(saveFileName, 'GenerateTime', 'TrainTime_LSTM', 'averageAccuracy_LSTM', 'stdAccuracy_LSTM', 'simNet_LSTM');
        fprintf('LSTM results saved to %s\n\n', saveFileName);
    end
end
end
end

%% Builders and helpers
%创建并添加神经网络的输入层
function [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize)
    inputName = 'input';
    inLayer = sequenceInputLayer(inputFeatureSize, 'Name', inputName);
    lgraph = addLayers(lgraph, inLayer);
end

%去噪残差块
%模型训练初期过滤信号中的噪声干扰，同时通过残差连接保留原始有效特征
%当原始特征（[160,4]）进入去噪块后，会同时输送到两个独立的分支
function [lgraph, outName] = addDenoiseBlock1D(lgraph, inName)
%主分支
    blk = [
        averagePooling1dLayer(3, 'Stride',1, 'Padding','same', 'Name','denoise_avg')%核大小 3×1，1D 平均池化
        convolution1dLayer(3, 8, 'Padding','same','Stride',1,'Name','denoise_conv')%1D 卷积：在去噪后提取局部时序特征，将通道数从 4 扩展到 8
        batchNormalizationLayer('Name','denoise_bn') %批量归一化
        reluLayer('Name','denoise_relu')
    ];
   %残差分支，将输入特征的通道数从4调整为8，确保能与主分支的输出（8 通道）进行加法合并
    matchConv = [
        convolution1dLayer(1, 8, 'Padding','same','Stride',1,'Name','denoise_match')
        batchNormalizationLayer('Name','denoise_match_bn')
    ];
    %融合主分支与残差分支特征
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
    stem = [
        convolution1dLayer(7, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'stem_conv')
        batchNormalizationLayer('Name', 'stem_bn')%批归一化层
                                                  %将输入数据转换为均值接近 0、标准差接近 1 的分布（或其他特定分布），避免数据分布过大或过小导致的训练问题
        reluLayer('Name', 'stem_relu')
        dropoutLayer(0.15, 'Name', 'stem_drop')%随机失活部分神经元，防止过拟合，轻微防止过拟合
    ];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, inputName, 'stem_conv');
    outName = 'stem_drop';
end

%普通残差块的定义
function [lgraph, outName] = addResidualBlock1D(lgraph, blockName, inChannels, outChannels, stride, inputName)
    mainLayers = [
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'Stride', stride, 'Name', blockName + "_conv1")
        batchNormalizationLayer('Name', blockName + "_bn1")
        reluLayer('Name', blockName + "_relu1")
        dropoutLayer(0.05, 'Name', blockName + "_drop1")
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
%普通ResNet模块
%分为恒等块和投影块
%恒等块不改变数据维度的情况下加深网络并提炼特征。
%投影块加深网络的同时，对数据进行下采样，缩短序列长度并改变通道数，增大感受野。
%由于主路径改变了数据的维度，短路连接的路径必须经过一个Conv 1x1模块来进行维度匹配，
function [lgraph, outName] = addResNetBackbone1D(lgraph, inName, embedDim)
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res1', 64, 64, 1, inName);%64, 64为输入/输出通道。步长为1，跳跃连接直接传递输入特征。
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res2', 64, 128, 2, outName);%输入通道64≠输出通道 128，步长为2（下采样）跳跃连接通过 1×1 卷积调整通道和分辨率。
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res3', 128, 128, 1, outName);%输入/输出通道均为 128，步长为1，跳跃连接直接传递输入特征。
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res4', 128, embedDim, 2, outName);%输入通道 128≠输出通道320，步长为2（下采样），跳跃连接通过 1×1 卷积调整。
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res5', embedDim, embedDim, 1, outName);%输入/输出通道均为320，步长为1，跳跃连接直接传递输入特征。
end

%空洞残差块的定义  在不降低序列长度的情况下，高效地扩大感受野，捕捉信号中长距离的时序依赖关系。
function [lgraph, outName] = addDilatedResidual1D(lgraph, blockName, inChannels, outChannels, dilation, inputName)
    mainLayers = [
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'DilationFactor', dilation, 'Stride', 1, 'Name', blockName + "_conv1")
        %关键的空洞卷积层。DilationFactor（扩张因子）参数让卷积核在计算时跳过一些输入点，扩大感受野。捕捉到时间上相距较远的特征之间的关联，而计算成本却没有增加。
        batchNormalizationLayer('Name', blockName + "_bn1")
        reluLayer('Name', blockName + "_relu1")
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'DilationFactor', 1, 'Stride', 1, 'Name', blockName + "_conv2") %标准的1D卷积层
        batchNormalizationLayer('Name', blockName + "_bn2")
    ];
    % 跳跃连接。通道不同调整通道数，通道相同，原始输入就直接连接到最后的加法层。
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
%空洞残差块 两个空洞残差块的堆叠 残差分支直接连接原始输入（因输入输出通道数相同）
function [lgraph, outName] = addDilatedStack1D(lgraph, inName, embedDim)
    [lgraph, outName] = addDilatedResidual1D(lgraph, 'dres1', embedDim, embedDim, 2, inName);  %dilation=2
    [lgraph, outName] = addDilatedResidual1D(lgraph, 'dres2', embedDim, embedDim, 4, outName); %dilation=4
end
%对齐块
%不是改变通道数量，而是对现有通道内的特征进行跨通道的线性组合与优化。
%为后续更复杂的Inception模块提供质量更高的输入。
function [lgraph, outName] = addAlignBlock1D(lgraph, inName, embedDim)
    alignBlock = [
        convolution1dLayer(1, embedDim, 'Padding', 'same', 'Stride', 1, 'Name', 'align_conv')
        batchNormalizationLayer('Name', 'align_bn')
        reluLayer('Name', 'align_relu')
    ];
    lgraph = addLayers(lgraph, alignBlock);
    lgraph = connectLayers(lgraph, inName, 'align_conv');
    outName = 'align_relu';
end
%Inception块定义
function [lgraph, outName] = addInceptionDilated1D(lgraph, inName, outChannels, blockId)

    br1 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br1_conv1'])%1×1 卷积先将320通道压缩至80通道
        reluLayer('Name',[blockId '_br1_relu1'])
        convolution1dLayer(3, outChannels/4, 'Padding','same','DilationFactor',1,'Name',[blockId '_br1_conv3'])%3×1卷积捕捉局部时序特征，序列长度始终保持 40
        batchNormalizationLayer('Name',[blockId '_br1_bn'])
        reluLayer('Name',[blockId '_br1_relu2'])
    ];
    br2 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br2_conv1'])
        reluLayer('Name',[blockId '_br2_relu1'])
        convolution1dLayer(3, outChannels/4, 'Padding','same','DilationFactor',2,'Name',[blockId '_br2_conv3d2']) %空洞率2的3×1卷积扩大感受野，
        batchNormalizationLayer('Name',[blockId '_br2_bn'])
        reluLayer('Name',[blockId '_br2_relu2'])
    ];
    br3 = [
        convolution1dLayer(1, outChannels/4, 'Padding','same','Stride',1,'Name',[blockId '_br3_conv1'])
        reluLayer('Name',[blockId '_br3_relu1'])
        convolution1dLayer(5, outChannels/4, 'Padding','same','DilationFactor',3,'Name',[blockId '_br3_conv5d3']) %5×1大卷积核+空洞率3捕捉长距离时序特征
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

    lgraph = connectLayers(lgraph, inName, [blockId '_br1_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br2_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br3_conv1']);
    lgraph = connectLayers(lgraph, inName, [blockId '_br4_pool']);

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

    addName = [blockId '_add'];
    lgraph = addLayers(lgraph, additionLayer(2,'Name',addName));
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
%时序卷积 卷积核更大5×1，捕捉的局部时序范围更广
function [lgraph, outName] = addTCNBlock1D(lgraph, inName, embedDim)
    [lgraph, name1] = addTCNUnit(lgraph, 'tcn1', embedDim, [1 2], 0.1, inName);
    [lgraph, name2] = addTCNUnit(lgraph, 'tcn2', embedDim, [2 4], 0.1, name1);
    [lgraph, outName] = addTCNUnit(lgraph, 'tcn3', embedDim, [4 8], 0.1, name2);
end
%时序卷积网络 双卷积层+空洞机制+残差连接 2 个不同空洞率
%通过空洞卷积扩大感受野，捕捉长距离的时序依赖关系
%时序卷积层定义
function [lgraph, outName] = addTCNUnit(lgraph, unitName, channels, dilations, dropoutP, inName)
    conv1 = convolution1dLayer(5, channels, 'Padding','same', 'DilationFactor', dilations(1), 'Name', unitName + "_conv1");%第1个空洞卷积
    bn1 = batchNormalizationLayer('Name', unitName + "_bn1");
    relu1 = reluLayer('Name', unitName + "_relu1");
    drop1 = dropoutLayer(dropoutP, 'Name', unitName + "_drop1");
    conv2 = convolution1dLayer(5, channels, 'Padding','same', 'DilationFactor', dilations(2), 'Name', unitName + "_conv2");%第2个空洞卷积
    bn2 = batchNormalizationLayer('Name', unitName + "_bn2");
    addL = additionLayer(2, 'Name', unitName + "_add");
    relu2 = reluLayer('Name', unitName + "_out");

    seq = [conv1; bn1; relu1; drop1; conv2; bn2];
    lgraph = addLayers(lgraph, seq);
    lgraph = addLayers(lgraph, addL);
    lgraph = addLayers(lgraph, relu2);
    %残差连接
    lgraph = connectLayers(lgraph, inName, unitName + "_conv1");
    lgraph = connectLayers(lgraph, unitName + "_bn2", unitName + "_add/in1");
    lgraph = connectLayers(lgraph, inName, unitName + "_add/in2");
    lgraph = connectLayers(lgraph, unitName + "_add", unitName + "_out");
    outName = unitName + "_out";
end

%BILSTM层定义
function [lgraph, outName] = addBiLSTMStackStrong(lgraph, inName)
    rnn = [
        bilstmLayer(256, 'OutputMode', 'sequence', 'Name', 'bilstm1') %第一个BILSTM
        dropoutLayer(0.3, 'Name', 'rnn_drop1')
        bilstmLayer(192, 'OutputMode', 'last', 'Name', 'bilstm2') %第二个BILSTM 仅输出最后一个时间步的数据，数据最终通道数为192+192
        dropoutLayer(0.3, 'Name', 'rnn_drop2')
    ];
    lgraph = addLayers(lgraph, rnn);
    lgraph = connectLayers(lgraph, inName, 'bilstm1');
    outName = 'rnn_drop2';
end
%分类输出层定义
function [lgraph, outName] = addClassifierHeadLS(lgraph, inName, numClasses, classNames, classWeights)
    head = [
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        dropoutLayer(0.35, 'Name', 'head_drop')
        fullyConnectedLayer(numClasses, 'Name', 'fc_final')
        softmaxLayer('Name', 'softmax')
    ];
    lgraph = addLayers(lgraph, head);
    if exist('LabelSmoothingClassificationLayer','class') == 8
        lsLayer = LabelSmoothingClassificationLayer(0.1, 'output');
        lgraph = addLayers(lgraph, lsLayer);
    else
        cls = classificationLayer('Name', 'output', 'Classes', classNames, 'ClassWeights', classWeights');
        lgraph = addLayers(lgraph, cls);
    end
    lgraph = connectLayers(lgraph, inName, 'fc1');
    lgraph = connectLayers(lgraph, 'softmax', 'output');
    outName = 'output';
end

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

function Z = normalizePerSequence(Z)
    for r = 1:size(Z,1)
        mu = mean(Z(r,:));
        Z(r,:) = Z(r,:) - mu;
        rmsv = sqrt(mean(Z(r,:).^2) + 1e-8);
        Z(r,:) = Z(r,:) / rmsv;
    end
end

function Z = augmentSequence(Z, SNR)
    T = size(Z,2);
    if SNR <= 0
        pMask = 0.40; pRot = 0.55; pGain = 0.45; pShift = 0.35; pNoise = 0.55; pFourier = 0.45;
        noiseStd = 0.06;
    else
        pMask = 0.30; pRot = 0.45; pGain = 0.35; pShift = 0.25; pNoise = 0.40; pFourier = 0.30;
        noiseStd = 0.04;
    end

    if rand < pMask && T > 24
        mlen = randi([8,22]);
        t0 = randi([1, max(1, T-mlen+1)]);
        Z(:, t0:min(T, t0+mlen-1)) = 0;
    end

    if rand < pRot
        theta = (pi/180) * (randn*5);
        R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        Z(1:2,:) = R * Z(1:2,:);
    end

    if rand < pGain
        g = 10^(randn*0.02);
        Z(1:3,:) = Z(1:3,:) * g;
    end

    if rand < pShift && T > 1
        s = randi([-3,3]);
        if s ~= 0
            Z = circshift(Z, [0 s]);
        end
    end

    if rand < pFourier
        for ch = 1:4
            x = Z(ch, :);
            X_fft = fft(x);
            mag = abs(X_fft);
            sorted_mag = sort(mag, 'descend');
            th = sorted_mag(max(1,round(0.25 * numel(sorted_mag)))) * 0.15;
            mask = ones(size(mag));
            mask(mag < th) = 0.6;
            Z(ch, :) = real(ifft(X_fft .* mask));
        end
    end

    if rand < pNoise
        Z = Z + noiseStd*randn(size(Z));
    end
end