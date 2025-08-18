classdef TemporalSelfAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
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

        function Z = forward(layer, X)
            % Ensure format to CTB (Channels x Time x Batch)
            Xctb = layer.format(X, 'CTB');

            % Sizes
            [channels, timeSteps, batchSize] = size(Xctb);

            % Flatten time and batch for linear projections
            X2 = reshape(Xctb, channels, []); % [C x (T*B)]

            % Linear projections
            Q = layer.Wq * X2 + layer.bq; % [U x (T*B)]
            K = layer.Wk * X2 + layer.bk; % [U x (T*B)]
            V = layer.Wv * X2 + layer.bv; % [U x (T*B)]

            % Reshape back to [U x T x B]
            U = layer.NumUnits;
            Q = reshape(Q, U, timeSteps, batchSize);
            K = reshape(K, U, timeSteps, batchSize);
            V = reshape(V, U, timeSteps, batchSize);

            % Attention scores: for each batch, (T x U) * (U x T) -> (T x T)
            Qt = permute(Q, [2 1 3]); % [T x U x B]
            Kt = permute(K, [1 2 3]); % [U x T x B]
            scores = pagemtimes(Qt, Kt) ./ sqrt(U); % [T x T x B]

            % Stable softmax over key/time dimension (dim=2)
            scores = scores - max(scores, [], 2);
            weights = exp(scores);
            weights = weights ./ sum(weights, 2);

            % Context: V * weights' -> [U x T x B]
            Y = pagemtimes(V, permute(weights, [2 1 3]));

            % Output projection
            Y2 = reshape(Y, U, []);
            O = layer.Wo * Y2 + layer.bo; % [U x (T*B)]
            O = reshape(O, U, timeSteps, batchSize);

            % Residual connection (if channel sizes match)
            if channels == U
                Out = O + Xctb;
            else
                Out = O;
            end

            % Restore original formatting
            Z = layer.formatInverse(Out);
        end
    end
end

