function y = mobius_add(u, v, c)
%MOBIUS_ADD Mobius addition.
    cuv = 1.0 + 2.0 * c * uv;
    c_norm_v_sq = c * norm(v)^2;
    norm_u_sq = norm(u)^2;

    a = (cuv + c_norm_v_sq) * u + (1.0 - c * norm_u_sq) * v;
    b = cuv + c * c_norm_v_sq * norm_u_sq;

    y = a / b;
end

