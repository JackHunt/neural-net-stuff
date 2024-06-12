function density = dp(alpha, K)
%DP Realises a K dimensional Dirichlet Process.
%   The realisation of the DP is performed by performing the so called
%   Stick Breaking Process. An alternative interpretation is the Chinese
%   Restaurant Process. The concentration of the DP is determined by
%   the alpha parameter.
    density = zeros(1, K);
    remaining_stick_length = 1;

    for i = 1:K
        break_fraction = betarnd(1, alpha);
        density(i) = remaining_stick_length * break_fraction;
        remaining_stick_length = remaining_stick_length - density(i);
    end

    density = sort(density, 'descend');
end
