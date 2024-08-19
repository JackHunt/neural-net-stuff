function [l, g, state] = modelLoss(net, loss_fn, X, T)
%MODELLOSS Evaluate model for training step.
    [Y, state] = forward(net, X);
    l = loss_fn(Y, T);
    g = dlgradient(l, net.Learnables);
end