function lp = gaussLogPrior(params, mu, sigma_sq)
%GAUSSLOGPRIOR Gaussian log prior.
    ln_Z = - 0.5 * numel(params) * log(2 * pi * sigma_sq);
    lp = -0.5 * sum((params - mu).^2) / sigma_sq + ln_Z;
end

