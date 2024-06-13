classdef DPDenseLayer < nnet.layer.Layer
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)
    properties
        InitAlpha
        K
    end

    properties (Learnable)
        W
        b
        alpha
    end

    methods
        function layer = DPDenseLayer(num_units, init_alpha)
            layer.InitAlpha = init_alpha;
            layer.K = num_units;
        end

        function layer = initialize(layer, layout)
            if isempty(layer.W)
                layer.W = heInit(layer.K, layout.Size(1));
            end
            
            if isempty(layer.b)
                layer.b = zeros([layer.K, 1], 'single');
            end

            if isempty(layer.alpha)
                layer.alpha = layer.InitAlpha;
            end
        end

        function Y = predict(layer, X)
            Y = layer.W * X + layer.b;
        end
    end
end