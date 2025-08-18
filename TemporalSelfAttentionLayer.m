classdef TemporalSelfAttentionLayer < nnet.layer.Layer
    % TemporalSelfAttentionLayer
    % Single-head scaled dot-product self-attention over time for sequence data.
    % Input/Output: same size as input [C x T x B] (channels x time x batch).

    properties (Learnable)
        Wq
        Wk
        Wv
        Wo
        bq
        bk
        bv
        bo
    end

    properties
        NumUnits (1,1) double {mustBePositive} = 256
    end

    methods
        function layer = TemporalSelfAttentionLayer(numUnits, name)
            % Constructor
            layer.Name = name;
            layer.NumUnits = numUnits;
            layer.Description = "Single-head temporal self-attention";

            % He-like initialization
            scale = sqrt(2/numUnits);
            layer.Wq = randn([numUnits numUnits]) * scale;
            layer.Wk = randn([numUnits numUnits]) * scale;
            layer.Wv = randn([numUnits numUnits]) * scale;
            layer.Wo = randn([numUnits numUnits]) * scale;
            layer.bq = zeros([numUnits 1]);
            layer.bk = zeros([numUnits 1]);
            layer.bv = zeros([numUnits 1]);
            layer.bo = zeros([numUnits 1]);
        end

        function Z = predict(layer, X)
            % X: [C x T x B]
            [channels, timeSteps, batchSize] = size(X);

            % Flatten time and batch for linear projections
            X2 = reshape(X, channels, []); % [C x (T*B)]

            % Linear projections
            Q = layer.Wq * X2 + layer.bq; % [U x (T*B)]
            K = layer.Wk * X2 + layer.bk; % [U x (T*B)]
            V = layer.Wv * X2 + layer.bv; % [U x (T*B)]

            % Reshape back to [U x T x B]
            U = layer.NumUnits;
            Q = reshape(Q, U, timeSteps, batchSize);
            K = reshape(K, U, timeSteps, batchSize);
            V = reshape(V, U, timeSteps, batchSize);

            % Attention scores per batch: (T x U) * (U x T) -> (T x T)
            Qt = permute(Q, [2 1 3]); % [T x U x B]
            Kt = K;                   % [U x T x B]
            scores = pagemtimes(Qt, Kt) ./ sqrt(U); % [T x T x B]

            % Stable softmax over time dimension (dim=2)
            scores = scores - max(scores, [], 2);
            weights = exp(scores);
            weights = weights ./ (sum(weights, 2) + eps);

            % Context: V * weights' -> [U x T x B]
            Y = pagemtimes(V, permute(weights, [2 1 3]));

            % Output projection
            Y2 = reshape(Y, U, []);
            O = layer.Wo * Y2 + layer.bo; % [U x (T*B)]
            O = reshape(O, U, timeSteps, batchSize);

            % Residual connection (if channel sizes match)
            if channels == U
                Z = O + X;
            else
                Z = O;
            end
        end

        function [dLdX, dLdWq, dLdbq, dLdWk, dLdbk, dLdWv, dLdbv, dLdWo, dLdbo] = backward(layer, X, Z, dLdZ, ~)
            % X:   [C x T x B]
            % Z:   [C x T x B]
            % dLdZ:[C x T x B]
            % Returns gradients w.r.t. input X and learnable params

            [channels, timeSteps, batchSize] = size(X);
            U = layer.NumUnits;

            % Forward recomputation of projections (needed for grads)
            X2 = reshape(X, channels, []); % [C x (T*B)]
            Q = layer.Wq * X2 + layer.bq; % [U x (T*B)]
            K = layer.Wk * X2 + layer.bk; % [U x (T*B)]
            V = layer.Wv * X2 + layer.bv; % [U x (T*B)]
            Q = reshape(Q, U, timeSteps, batchSize);
            K = reshape(K, U, timeSteps, batchSize);
            V = reshape(V, U, timeSteps, batchSize);

            % Compute weights as in predict for consistency
            Qt = permute(Q, [2 1 3]); % [T x U x B]
            scores = pagemtimes(Qt, K) ./ sqrt(U); % [T x T x B]
            scores = scores - max(scores, [], 2);
            weights = exp(scores);
            weights = weights ./ (sum(weights, 2) + eps);

            % Gradient starts at Z = O + X -> dLdO = dLdZ, dLdX accumulates dLdZ (residual)
            dLdO = dLdZ; % [C x T x B]

            % Output projection O = Wo * Y2 + bo, where Y = V * weights'
            dLdO2 = reshape(dLdO, U, []); % [U x (T*B)]

            % Recompute Y for use in gradients
            Y = pagemtimes(V, permute(weights, [2 1 3])); % [U x T x B]
            Y2 = reshape(Y, U, []);

            % Param grads for output projection
            dLdWo = dLdO2 * Y2.';
            dLdbo = sum(dLdO2, 2);
            dLdY2 = layer.Wo.' * dLdO2; % [U x (T*B)]
            dLdY = reshape(dLdY2, U, timeSteps, batchSize); % [U x T x B]

            % Y = V * weights': per batch b
            % dLdV_b = dLdY_b * weights_b
            % dLdWeights_b = (dLdY_b') * V_b
            dLdV = pagemtimes(dLdY, weights); % [U x T x B]
            dLdW = pagemtimes(permute(dLdY, [2 1 3]), permute(V, [1 2 3])); % [T x T x B]

            % Backprop through softmax: weights = softmax(scores) row-wise over dim=2
            % For each row i: dL/dscores_i = (dL/dw_i .* w_i) - w_i * sum(dL/dw_i .* w_i)
            tmp = dLdW .* weights;                 % [T x T x B]
            sumRow = sum(tmp, 2);                  % [T x 1 x B]
            dLdScores = tmp - weights .* sumRow;   % [T x T x B]

            % scores = Qt * K / sqrt(U)
            dLdQt = pagemtimes(dLdScores, permute(K, [2 1 3])) ./ sqrt(U); % [T x U x B]
            dLdK  = pagemtimes(permute(Qt, [2 1 3]), dLdScores) ./ sqrt(U); % [U x T x B]
            dLdQ  = permute(dLdQt, [2 1 3]); % [U x T x B]

            % V contribution already in dLdV

            % Combine grads back to X via projections
            dLdQ2 = reshape(dLdQ, U, []); % [U x (T*B)]
            dLdK2 = reshape(dLdK, U, []);
            dLdV2 = reshape(dLdV, U, []);

            dLdWq = dLdQ2 * X2.';
            dLdbq = sum(dLdQ2, 2);
            dLdX_from_Q = layer.Wq.' * dLdQ2; % [C x (T*B)]

            dLdWk = dLdK2 * X2.';
            dLdbk = sum(dLdK2, 2);
            dLdX_from_K = layer.Wk.' * dLdK2;

            dLdWv = dLdV2 * X2.';
            dLdbv = sum(dLdV2, 2);
            dLdX_from_V = layer.Wv.' * dLdV2;

            dLdX_proj = dLdX_from_Q + dLdX_from_K + dLdX_from_V; % [C x (T*B)]
            dLdX_proj = reshape(dLdX_proj, channels, timeSteps, batchSize);

            % Residual add from Z = O + X
            dLdX = dLdX_proj + dLdZ;
        end
    end
end

