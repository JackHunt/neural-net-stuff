classdef DPDropout < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        InitAlpha
        InitCDFCutoff
    end

    properties (Learnable)
        alpha
        cdf_cutoff
    end

    methods
        function layer = DPDropout(init_alpha, init_cdf_cutoff, num_output, name)
            layer.InitAlpha = init_alpha;
            layer.InitCDFCutoff = init_cdf_cutoff;
            layer.NumOutputs = num_output;
            layer.Name = name;
        end

        function layer = initialize(layer, ~)
            layer.alpha = layer.InitAlpha;
            layer.cdf_cutoff = layer.InitCDFCutoff;
        end
        
        function Y = predict(layer, X)
            Y = layer.forward(X);
        end

        function Y = forward(layer, X)
            % Make a draw from the DP via stick breaking process.
            density = dp(layer.alpha, layer.NumOutputs);
            [cdf, idx] = cdfCutoffIndex(density, layer.cdf_cutoff);

            % Generate mask.
            %B = size(X, 4);
        end
    end
end