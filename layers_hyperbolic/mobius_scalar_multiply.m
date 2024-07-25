function y = mobius_scalar_multiply(r, v, c)
%MOBIUS_SCALAR_MULTIPLY Mobius scalar multiplication.
    norm_v = norm(v);
    sqrt_c = sqrt(c);
    y = tanh(r * atanh(sqrt_c * norm_v)) * v / (sqrt_c * norm_v);
end

