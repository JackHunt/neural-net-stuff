classdef MobiusDense < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    properties
        D
    end

    properties (Learnable)
        W
        b
    end

    methods
        function layer = MobiusDense(num_units)
            layer.D = num_units;
        end

        function layer = initialize(layer, layout)
            N = prod(layout.Size(1:3));

            if isempty(layer.W)
                layer.W = dlarray(heInit(layer.D, N));
            end
            
            if isempty(layer.b)
                layer.b = dlarray(zeros([layer.D, 1], 'single'));
            end
        end

        function Y = predict(layer, X)
            %
        end
    end
end