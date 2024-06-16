function [x_train, y_train, x_val, y_val] = trainValSplit(x, y, train_percent)
%TRAINVALSPLIT Split data into training and validation sets.
%   TODO make this indices only so shape need not be known.
    n = size(x, 4);
    idx = randperm(n);
    val_start = floor(train_percent * n);

    train_idx = idx(1:val_start-1);
    val_idx = idx(val_start:end);

    x_train = x(:, :, :, train_idx);
    y_train = y(train_idx, :);

    x_val = x(:, :, :, val_idx);
    y_val = y(val_idx, :);
end

