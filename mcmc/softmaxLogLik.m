function ll = softmaxLogLik(net, X, Y)
%SOFTMAXLOGLIK Softmax log likelihood.
    dlX = dlarray(X', 'CB'); % Convert to dlarray
    
    Y_pred = predict(net, dlX); % Predict using the network
    Y_pred = softmax(Y_pred);
    Y_pred = gather(extractdata(Y_pred))';
    
    ll = sum(log(Y_pred(sub2ind(size(Y_pred), (1:size(Y_pred, 1))', Y))));
end

