function W = heInit(rows, cols)
%HEINIT Generate a random, He initialized matrix.
    sd = sqrt(2 / cols);
    W = sd * randn(rows, cols, 'single');
end