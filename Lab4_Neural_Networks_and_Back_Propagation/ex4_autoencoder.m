%% Generate dataset
load data/Iris.txt
data = Iris;
x = Iris(:, 1 : end - 1);
neuron_num = [4 3 4];

%% Train autoencoder
train_prop = 0.8;
learning_rate = 1.0;
learning_rate_decay = 0.99;
lambda = 0.00;                   % Weight Decay
epoch_num = 10000;

% Normalize dataset
maxX = max(x);
minX = min(x);
meanX = mean(x);
stdX = std(x);
y = (x - minX) ./ (maxX - minX);
x = (x - mean(x)) ./ std(x);

% Separate dataset
train_x = x(1 : int32(size(x, 1) * 0.8), :);
test_x = x(int32(size(x, 1) * 0.8) + 1 : end, :);
train_y = y(1 : int32(size(y, 1) * 0.8), :);
test_y = y(int32(size(y, 1) * 0.8) + 1 : end, :);

train_nn(train_x, train_y, test_x, test_y, neuron_num, learning_rate, epoch_num, learning_rate_decay, lambda, maxX, minX, meanX, stdX);