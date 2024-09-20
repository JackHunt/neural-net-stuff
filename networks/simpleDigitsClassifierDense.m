function layers = simpleDigitsClassifierDense()
%SIMPLEDIGITSCLASSIFIERDENSE A simple MNIST MLP.
    layers = [
        imageInputLayer([28 28 1])
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(10)
        % softmaxLayer
    ];
end

