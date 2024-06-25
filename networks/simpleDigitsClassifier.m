function layers = simpleDigitsClassifier()
%SIMPLEDIGITSCLASSIFIER A simple MNIST convnet taken from the docs.
    layers = [
        % Block 1
        imageInputLayer([28 28 1])
        convolution2dLayer(3, 8, Padding="same")
        batchNormalizationLayer
        reluLayer
        averagePooling2dLayer(2, Stride=2)
        % Block 2
        convolution2dLayer(3, 16, Padding="same")
        batchNormalizationLayer
        reluLayer
        averagePooling2dLayer(2, Stride=2)
        % Block 3
        convolution2dLayer(3, 32, Padding="same")
        batchNormalizationLayer
        reluLayer
        % Block 4
        convolution2dLayer(3, 32, Padding="same")
        batchNormalizationLayer
        reluLayer
        % Block 5
        %fullyConnectedLayer(10)
        DPDenseLayer(10, 0.1)
        softmaxLayer
    ];
end

