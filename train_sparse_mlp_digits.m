%% Clear env.
clear all;
close all;
clc;

addpath('data');
addpath('networks');
addpath('util')

%% Config options.
train_split = 0.8;
bs = 64;
lr = 1e-4;
momentum = 0.9;
epochs = 300;
init_dropout_rate = 0.1;
final_dropout_rate = 0.9;

%% Load dataset & compute train/val split.
digits = loadDIGITS();

[x, y, x_val, y_val] = trainValSplit( ...
    digits.x_train, digits.y_train, train_split);

xy = combine( ...
    arrayDatastore(x, IterationDimension=4), arrayDatastore(y));

xy_val = combine( ...
    arrayDatastore(x_val, IterationDimension=4), arrayDatastore(y_val));

%% Setup network and training options.
layers = sparseDigitsClassifier([512, 512, 512], 0.1);

val_freq = floor(size(x, 4) / bs);

mbq = minibatchqueue(xy, ...
    'MiniBatchSize', bs, ...
    'MiniBatchFormat', ["SSBC" "BC"], ...
    'OutputAsDlarray', true, ...
    'PartialMiniBatch', 'discard');

iters_per_epoch = floor(digits.n_train / bs);
num_iters = epochs * iters_per_epoch;

monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch", "DropoutRate"], ...
    XLabel="Iteration");

%% Train.
net = dlnetwork(layers);
net = initialize(net);

loss_fn = @(y_pred, y_true) crossentropy(y_pred, y_true, ...
    'ClassificationMode','multilabel');

epoch = 1;
velocity = [];
updated_dropout_rate = init_dropout_rate;
while epoch < epochs && ~monitor.Stop
    shuffle(mbq);

    epoch_loss = 0;
    iter = 0;
    while hasdata(mbq) && ~monitor.Stop
        [X, T] = next(mbq);

        [l, g, state] = dlfeval(@modelLoss, net, loss_fn, X, T);
        net.State = state;
        epoch_loss = epoch_loss + l;

        [net, velocity] = sgdmupdate(net, g, velocity, lr, momentum);

        iter = iter + 1;
    end

    epoch_loss = epoch_loss / iter;

    recordMetrics(monitor, epoch, Loss=epoch_loss);
    updateInfo(monitor, Epoch=epoch, DropoutRate=updated_dropout_rate);
    monitor.Progress = (epoch / epochs) * 100;
    
    % Update dropout rate.
    updated_dropout_rate = init_dropout_rate + ...
        (final_dropout_rate - init_dropout_rate) * (epoch / epochs);
    for i = 1:length(net.Layers)
        if isa(net.Layers(i), 'nnet.cnn.layer.DropoutLayer')
            net = updateDropoutRate(net, i, updated_dropout_rate);
        end
    end
    
    epoch = epoch + 1;
end