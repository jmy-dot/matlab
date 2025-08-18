classdef TemporalSelfAttentionLayer < nnet.layer.Layer
    % TemporalSelfAttentionLayer
    % Single-head scaled dot-product self-attention over time for sequence data.
    % Supports inputs [C x T x B] or [T x C x B]. Internally uses [U x T x B].

    properties (Learnable)
        Wq   % [U x U]
        Wk   % [U x U]
        Wv   % [U x U]
        Wo   % [U x U]
        bq   % [U x 1]
        bk   % [U x 1]
        bv   % [U x 1]
        bo   % [U x 1]
    end

    properties
        NumUnits (1,1) double {mustBePositive} = 256
    end

    methods
        function layer = TemporalSelfAttentionLayer(numUnits, name)
            layer.Name = name;
            layer.NumUnits = numUnits;
            layer.Description = "Single-head temporal self-attention";

            U = numUnits;
            scale = sqrt(2/numUnits);
            layer.Wq = randn([U U]) * scale;
            layer.Wk = randn([U U]) * scale;
            layer.Wv = randn([U U]) * scale;
            layer.Wo = randn([U U]) * scale;
            layer.bq = zeros([U 1]);
            layer.bk = zeros([U 1]);
            layer.bv = zeros([U 1]);
            layer.bo = zeros([U 1]);
        end

        function Z = predict(layer, X)
            U = layer.NumUnits;
            % Normalize input to [U x T x B]
            [Xutb, needPermute] = iToUTB(X, U);
            [~, T, B] = size(Xutb);

            % Linear projections on flattened time-batch
            X2 = reshape(Xutb, U, []); % [U x (T*B)]
            Q = layer.Wq * X2 + layer.bq; % [U x (T*B)]
            K = layer.Wk * X2 + layer.bk; % [U x (T*B)]
            V = layer.Wv * X2 + layer.bv; % [U x (T*B)]

            Q = reshape(Q, U, T, B);
            K = reshape(K, U, T, B);
            V = reshape(V, U, T, B);

            % scores: [T x T x B]
            Qt = permute(Q, [2 1 3]); % [T x U x B]
            scores = pagemtimes(Qt, K) ./ sqrt(U);
            scores = scores - max(scores, [], 2);
            W = exp(scores); W = W ./ (sum(W, 2) + eps);

            % Context and output projection
            Y = pagemtimes(V, permute(W, [2 1 3])); % [U x T x B]
            Y2 = reshape(Y, U, []);
            O = layer.Wo * Y2 + layer.bo; % [U x (T*B)]
            O = reshape(O, U, T, B);

            % Residual
            Zutb = O + Xutb;
            Z = iFromUTB(Zutb, needPermute);
        end

        function [dLdX, dLdWq, dLdWk, dLdWv, dLdWo, dLdbq, dLdbk, dLdbv, dLdbo] = backward(layer, X, Z, dLdZ, ~)
            U = layer.NumUnits;
            % Normalize to [U x T x B]
            [Xutb, needPermute] = iToUTB(X, U);
            dLdZutb = iToUTB(dLdZ, U);
            [~, T, B] = size(Xutb);

            % Forward recompute
            X2 = reshape(Xutb, U, []);
            Q2 = layer.Wq * X2 + layer.bq; K2 = layer.Wk * X2 + layer.bk; V2 = layer.Wv * X2 + layer.bv;
            Q = reshape(Q2, U, T, B); K = reshape(K2, U, T, B); V = reshape(V2, U, T, B);
            Qt = permute(Q, [2 1 3]);
            scores = pagemtimes(Qt, K) ./ sqrt(U);
            scores = scores - max(scores, [], 2);
            W = exp(scores); W = W ./ (sum(W, 2) + eps);

            % Output projection grads
            dLdO2 = reshape(dLdZutb, U, []);
            Y = pagemtimes(V, permute(W, [2 1 3]));
            Y2 = reshape(Y, U, []);
            dLdWo = dLdO2 * Y2.';      % [U x U]
            dLdbo = sum(dLdO2, 2);
            dLdY2 = layer.Wo.' * dLdO2; % [U x (T*B)]
            dLdY  = reshape(dLdY2, U, T, B);

            % dLdV, dLdW
            dLdV = pagemtimes(dLdY, W); % [U x T x B]
            dLdW = pagemtimes(permute(dLdY, [2 1 3]), permute(V, [1 2 3])); % [T x T x B]

            % Backprop softmax
            tmp = dLdW .* W; sumRow = sum(tmp, 2);
            dLdScores = tmp - W .* sumRow;

            % To Q,K
            dLdQt = pagemtimes(dLdScores, permute(K, [2 1 3])) ./ sqrt(U); % [T x U x B]
            dLdK  = pagemtimes(permute(Qt, [2 1 3]), dLdScores) ./ sqrt(U); % [U x T x B]
            dLdQ  = permute(dLdQt, [2 1 3]); % [U x T x B]

            % Flatten
            dLdQ2 = reshape(dLdQ, U, []);
            dLdK2 = reshape(dLdK, U, []);
            dLdV2 = reshape(dLdV, U, []);

            % Param grads fixed to [U x U]
            dLdWq = dLdQ2 * X2.';   % [U x U]
            dLdWk = dLdK2 * X2.';   % [U x U]
            dLdWv = dLdV2 * X2.';   % [U x U]
            dLdbq = sum(dLdQ2, 2);
            dLdbk = sum(dLdK2, 2);
            dLdbv = sum(dLdV2, 2);

            % Input grads
            dLdX_from_Q = layer.Wq.' * dLdQ2; % [U x (T*B)]
            dLdX_from_K = layer.Wk.' * dLdK2;
            dLdX_from_V = layer.Wv.' * dLdV2;
            dLdX2 = dLdX_from_Q + dLdX_from_K + dLdX_from_V;
            dLdXutb = reshape(dLdX2, U, T, B) + dLdZutb; % residual

            % Restore input orientation
            dLdX = iFromUTB(dLdXutb, needPermute);
        end
    end
end

% Utilities
function [Xutb, needPermute] = iToUTB(X, U)
% Return data as [U x T x B]. If X is [T x U x B], transpose to [U x T x B].
    sz = size(X);
    if numel(sz) < 3, sz(3) = 1; end
    if sz(1) == U
        Xutb = X; needPermute = false;
    elseif sz(2) == U
        Xutb = permute(X, [2 1 3]); needPermute = true;
    else
        error('TemporalSelfAttentionLayer:InvalidSize', 'Expected size(X,1) or size(X,2) to equal NumUnits=%d, got [%s].', U, num2str(sz));
    end
end

function X = iFromUTB(Xutb, needPermute)
% Convert back to original orientation
    if needPermute
        X = permute(Xutb, [2 1 3]);
    else
        X = Xutb;
    end
end

