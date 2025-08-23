classdef FourierDenoiseLayer < nnet.layer.Layer
    % FourierDenoiseLayer - Custom layer for frequency domain denoising
    
    properties
        % Learnable parameters
        FrequencyThreshold (1,1) double = 0.1  % Frequency threshold for noise removal
        NoiseReductionFactor (1,1) double = 0.5  % Noise reduction factor
    end
    
    methods
        function layer = FourierDenoiseLayer(name)
            % Constructor
            layer.Name = name;
            layer.Description = 'Fourier domain denoising layer';
        end
        
        function Z = predict(layer, X)
            % Forward pass: Apply FFT, denoise in frequency domain, then IFFT
            % X: [channels x time x batch]
            
            [numChannels, numTime, numBatch] = size(X);
            Z = zeros(size(X));
            
            for b = 1:numBatch
                for c = 1:numChannels
                    % Extract time series for current channel and batch
                    x = X(c, :, b);
                    
                    % Apply FFT
                    X_fft = fft(x);
                    
                    % Get magnitude spectrum
                    magnitude = abs(X_fft);
                    
                    % Find noise threshold (adaptive based on spectrum)
                    sorted_mag = sort(magnitude, 'descend');
                    noise_threshold = sorted_mag(round(0.3 * length(sorted_mag))) * layer.FrequencyThreshold;
                    
                    % Create frequency mask for denoising
                    % Keep strong frequency components, attenuate weak ones
                    mask = ones(size(magnitude));
                    weak_indices = magnitude < noise_threshold;
                    mask(weak_indices) = layer.NoiseReductionFactor;
                    
                    % Apply mask to frequency domain
                    X_fft_denoised = X_fft .* mask;
                    
                    % Apply IFFT to get denoised signal
                    x_denoised = real(ifft(X_fft_denoised));
                    
                    % Store result
                    Z(c, :, b) = x_denoised;
                end
            end
        end
        
        function [dLdX, dLdW] = backward(layer, X, Z, dLdZ, memory)
            % Backward pass: Compute gradients
            % For simplicity, we'll use a simplified gradient approximation
            % In practice, you might want to implement a more sophisticated gradient
            
            [numChannels, numTime, numBatch] = size(X);
            dLdX = zeros(size(X));
            
            for b = 1:numBatch
                for c = 1:numChannels
                    % Extract time series
                    x = X(c, :, b);
                    dz = dLdZ(c, :, b);
                    
                    % Apply FFT
                    X_fft = fft(x);
                    magnitude = abs(X_fft);
                    
                    % Find noise threshold
                    sorted_mag = sort(magnitude, 'descend');
                    noise_threshold = sorted_mag(round(0.3 * length(sorted_mag))) * layer.FrequencyThreshold;
                    
                    % Create gradient mask
                    mask = ones(size(magnitude));
                    weak_indices = magnitude < noise_threshold;
                    mask(weak_indices) = layer.NoiseReductionFactor;
                    
                    % Apply mask to gradient
                    dLdX(c, :, b) = dz .* mask(1:numTime);
                end
            end
            
            % No learnable parameters in this layer
            dLdW = [];
        end
    end
end