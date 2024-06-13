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

[x, y, x_val, y_val] = trainValSplit(digits.x_train, digits.y_train, 0.85);

%% Setup network and training options.
layers = simpleDigitsClassifier();

bs  = 128;
lr = 1e-3;
val_freq = floor(numel(x) / bs);

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
net = trainnet(x, y, layers, "mse", options);