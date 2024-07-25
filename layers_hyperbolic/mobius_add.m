function y = mobius_add(u, v, c)
%MOBIUS_ADD Mobius addition.
    cuv = 1.0 + 2.0 * c * uv;
    c_norm_v2 = c * norm(v)^2;
    norm_u2 = norm(u)^2;

    a = (cuv + c_norm_v2) * u + (1.0 - c * norm_u2) * v;
    b = cuv + c * c_norm_v2 * norm_u2;

    y = a / b;
end

