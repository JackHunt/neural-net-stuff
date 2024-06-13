%% Clear & setup paths.
close all;
clear variables;
clc;

addpath('data');
addpath('networks');

display_sample = false;

%% Load dataset & compute train/val split.
digits = loadDIGITS();

if display_sample
    idx = randperm(digits.n_train, 49);
    I = imtile(digits.x_train(:, :, :, idx));
    figure
    imshow(I);
end

[train_idx, val_idx] = trainingPartitions(digits.n_train,[0.85 0.15]);

x_train = digits.x_train(:, :, :, train_idx);
y_train = digits.y_train(train_idx);

x_val = digits.x_train(:, :, :, val_idx);
y_val = digits.y_train(val_idx);

%% Setup network architecture.
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
    fullyConnectedLayer(10)
];

%% Setup training options.
bs  = 128;
lr = 1e-3;
val_freq = floor(numel(x_train) / bs);

options = trainingOptions("sgdm", ...
    MiniBatchSize=bs, ...
    InitialLearnRate=lr, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=20, ...
    Shuffle="every-epoch", ...
    ValidationData={x_val, y_val}, ...
    ValidationFrequency=val_freq, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=false);

%% Train
net = trainnet(x_train, y_train, layers, "mse", options);