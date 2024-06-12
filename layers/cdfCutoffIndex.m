function [cdf, idx] = cdfCutoffIndex(density, cutoff_p)
%CDFCUTOFFINDEX Find the index of a density CDF up to a CDF value.
%   Takes a (sorted) probability density and a probability threshold and
%   computes the CDF (via its cumulative sum) and the index of the max
%   CDF value that does not exceed the provided cutoff value.
    cdf = cumsum(density);
    idx = find(cdf <= cutoff_p, 1, 'last');
end
