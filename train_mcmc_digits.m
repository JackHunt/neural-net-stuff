%% Clear & setup paths.
close all;
clear variables;
clc;

addpath('data');
addpath('networks');
addpath('layers');

%% Config options.
train_split = 0.8;

%% Load dataset & compute train/val split.
digits = loadDIGITS();

[x, y, x_val, y_val] = trainValSplit(digits.x_train, digits.y_train, train_split);

%% Setup network and training options.
layers = simpleDigitsClassifierDense();

net = dlnetwork(layers);