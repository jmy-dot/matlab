classdef LabelSmoothingClassificationLayer < nnet.layer.ClassificationLayer
    % LabelSmoothingClassificationLayer - Cross-entropy with label smoothing
    % Usage: layer = LabelSmoothingClassificationLayer(epsilon, name)

    properties
        Epsilon (1,1) double {mustBeGreaterThanOrEqual(Epsilon,0), mustBeLessThan(Epsilon,1)} = 0.1
    end

    methods
        function layer = LabelSmoothingClassificationLayer(epsilon, name)
            layer.Name = name;
            layer.Description = sprintf('Label-smoothed cross-entropy (epsilon=%.3f)', epsilon);
            layer.Epsilon = epsilon;
        end

        function loss = forwardLoss(layer, Y, T)
            % Y: dlarray [numClasses x batch]
            % T: one-hot targets [numClasses x batch]
            eps = layer.Epsilon;
            numClasses = size(Y,1);
            % Smooth targets
            Tsm = (1 - eps) * T + eps/numClasses;
            % Clip to avoid log(0)
            Y = min(max(Y, 1e-7), 1 - 1e-7);
            % Cross-entropy
            lossPerSample = -sum(Tsm .* log(Y), 1);
            loss = mean(lossPerSample, 'all');
        end
    end
end