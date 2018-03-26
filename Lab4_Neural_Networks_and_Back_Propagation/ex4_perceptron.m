%% Load data
load data/Iris.txt
data = Iris;

% Randomly Shuffle dataset
data = data(randperm(size(data, 1)), :);

% Separate into x and y data
x = data(:, 1 : end - 1)';
t1 = data(:, end)';
t2 = t1;

% t1 sets the first class as 1, others 0
t1(t1 ~= 1) = 0;
% t2 sets the second class as 1, others 0
t2(t2 ~= 2) = 0;
t2(t2 == 2) = 1;

%% Train perceptron
accuracy = 0;

net1 = perceptron;
net1 = train(net, x, t1);
view(net1)
y1 = net1(x);

net2 = perceptron;
net2 = train(net, x, t2);
y2 = net2(x);

sum()