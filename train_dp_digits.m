%% Clear & setup paths.
close all;
clear variables;
clc;

addpath('data');
addpath('networks');
addpath('layers');

%% Config options.
display_sample = false;
train_split = 0.85;
bs = 64;
lr = 1e-3;
epochs = 30;

%% Load dataset & compute train/val split.
digits = loadDIGITS();

if display_sample
    idx = randperm(digits.n_train, 49);
    I = imtile(digits.x_train(:, :, :, idx));
    figure
    imshow(I);
end

[x, y, x_val, y_val] = trainValSplit(digits.x_train, digits.y_train, train_split);

%% Setup network and training options.
layers = simpleDigitsClassifier();

val_freq = floor(size(x, 4) / bs);

options = trainingOptions("sgdm", ...
    MiniBatchSize=bs, ...
    InitialLearnRate=lr, ...
    MaxEpochs=epochs, ...
    Shuffle="every-epoch", ...
    ValidationData={x_val, y_val}, ...
    ValidationFrequency=val_freq, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

%% Train.
net = trainnet(x, y, layers, "crossentropy", options);