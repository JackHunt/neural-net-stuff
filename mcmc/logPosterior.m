function p = logPosterior(lik_fn, prior_fn)
%LOGPOSTERIOR A generic log posterior - assumed to be zero arg lambdas.
    p = lik_fn() + prior_fn();
end

