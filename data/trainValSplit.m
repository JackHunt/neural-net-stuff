function [x_train, y_train, x_val, y_val] = trainValSplit(x, y, train_percent)
%TRAINVALSPLIT Summary of this function goes here
%   Detailed explanation goes here
    train_idx = 0;
    val_idx = 0;

    x_train = x(:, :, :, train_idx);
    y_train = y(train_idx);

    x_val = x(:, :, :, val_idx);
    y_val = y(val_idx);
end

