function layers = mlpWithDropoutReLU(layer_dims, dropout_rate)
%MLPWITHDROPOUTRELU An MLP with dropout applied at each layer.
    layers = [];
    for n = 1 : numel(layer_dims)
        layers = [
            layers
            fullyConnectedLayer(layer_dims(n))
            reluLayer
            dropoutLayer(dropout_rate)
        ];
    end
end

