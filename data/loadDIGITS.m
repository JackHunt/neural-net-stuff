function digits = loadDIGITS(normalise)
%LOADDIGITS Loads the digits (MNIST like) dataset.
arguments
    normalise = true;
end
    [x_train, y_train, ~] = digitTrain4DArrayData;
    [x_test, y_test, ~] = digitTest4DArrayData;

    if normalise
        b = 255;
    else
        b = 1;
    end

    digits.x_train = x_train / b;
    digits.y_train = onehotencode(categorical(y_train), 2);
    digits.n_train = size(digits.x_train, 4);
    
    digits.x_test = x_test / b;
    digits.y_test = onehotencode(categorical(y_test), 2);
    digits.n_test = size(digits.x_test, 4);
end

