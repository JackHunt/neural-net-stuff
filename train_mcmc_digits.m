%% Clear & setup paths.
close all;
clear variables;
clc;

addpath('data');
addpath('mcmc')
addpath('networks');
addpath('layers');

%% Config options.
train_split = 0.8;

%% Load dataset & compute train/val split.
digits = loadDIGITS();

[X, Y, X_val, Y_val] = trainValSplit(digits.x_train, digits.y_train, train_split);

%% Setup network and training options.
% Network architecture.
layers = simpleDigitsClassifierDense();
net = dlnetwork(layers);

%% Setup log PDFs.
% Likelihood & prior.
log_lik = softmaxLogLik;
log_prior = @(params) gaussLogPrior(params, 0, 1);

% Posterior.
log_posterior = @(params) logPosterior(params, net, X, Y, log_lik, log_prior);

% Proposal distribution.
proposal = @(params) params + 0.1 * randn(size(params));

%% Run MCMC
params_init = extractParameters(net);

num_samples = 1000;
samples = mhsample(params_init, num_samples, 'logpdf', log_posterior, 'proprnd', proposal);