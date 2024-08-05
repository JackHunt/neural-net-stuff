function p = logPosterior(params, net, X, Y, log_lik_fn, log_prior_fn)
%LOGPOSTERIOR A generic log posterior
    net = updateParameters(net, params);
    ll = log_lik_fn(net, X, Y);
    lp = log_prior_fn(params);
    p = ll + lp;
end

