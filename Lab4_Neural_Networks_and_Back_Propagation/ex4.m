%% Set Parameters and load data
load data/Iris.txt
data = Iris;
neuron_num = [4, 5, 3];
% load data/Pima_Indians.txt
% data = Pima_Indians;
% neuron_num = [8, 12, 12, 1];
% load data/wdbc.txt
% data = wdbc;
% neuron_num = [30, 30, 1];
train_prop = 0.8;
learning_rate = 0.8;
learning_rate_decay = 1.00;     % Learning rate decay after every 100 iters
lambda = 0.00;                   % Weight Decay
epoch_num = 30000;
softmax = false;
if(neuron_num(end) > 1)
    softmax = true;
end

%% Preprocess data
% Randomly Shuffle dataset
data = data(randperm(size(data, 1)), :);
x = data(:, 1 : end - 1);
y = data(:, end);
% Regularize x data
x = (x - mean(x)) ./ std(x);
% Transform to multivariate form
if(softmax)
    y_new = zeros(length(y), neuron_num(end));
    for i = 1 : length(y)
        y_new(i, y(i)) = 1;
    end
    y = y_new;
end
% Separate into train and test data
train_x = x(1 : int32(size(x, 1) * 0.8), :);
test_x = x(int32(size(x, 1) * 0.8) + 1 : end, :);
train_y = y(1 : int32(size(y, 1) * 0.8), :);
test_y = y(int32(size(y, 1) * 0.8) + 1 : end, :);

%% Train neural network
train_nn(train_x, train_y, test_x, test_y, neuron_num, learning_rate, epoch_num, learning_rate_decay, lambda);