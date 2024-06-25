classdef DPDenseLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ... % (Optional) 
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
            N = prod(layout.Size(1:3));

            if isempty(layer.W)
                layer.W = dlarray(heInit(layer.K, N));
            end
            
            if isempty(layer.b)
                layer.b = dlarray(zeros([layer.K, 1], 'single'));
            end

            if isempty(layer.alpha)
                layer.alpha = dlarray(layer.InitAlpha);
            end
        end

        function Y = predict(layer, X)
            p = dp(layer.alpha, layer.K);
            Y = fullyconnect(X, layer.W, layer.b) * p;
        end
    end
end