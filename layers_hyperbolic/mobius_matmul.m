function y = mobius_matmul(M, X, c)
%MOBIUS_MATMUL Mobius matrix multiplication.
    MX = M * X;
    MX_norm = norm(MX);
    X_norm = norm(X);
    sqrt_c = sqrt(c);

    a = 1 / sqrt_c * tanh(MX_norm / X_norm * atanh(sqrt_c * X_norm));
    b = MX_norm * MX;

    y = a / b;
end

