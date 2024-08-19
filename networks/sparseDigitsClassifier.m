function layers = sparseDigitsClassifier(layer_dims, sparsity)
%SPARSEDIGITSCLASSIFIER Sparse, fully MLP digits classifier.
    backbone = mlpWithDropoutReLU(layer_dims, sparsity);
    layers = [
        imageInputLayer([28 28 1])
        flattenLayer
        backbone
        fullyConnectedLayer(10)
    ];
end

