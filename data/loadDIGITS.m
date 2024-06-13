function digits = loadDIGITS()
%LOADDIGITS Loads the digits (MNIST like) dataset.
    [x_train, y_train, ~] = digitTrain4DArrayData;
    [x_test, y_test, ~] = digitTest4DArrayData;

    digits.x_train = x_train;
    digits.y_train = onehotencode(categorical(y_train), 2);
    digits.n_train = size(digits.x_train, 4);
    
    digits.x_test = x_test;
    digits.y_test = onehotencode(categorical(y_test), 2);
    digits.n_test = size(digits.x_test, 4);
end

