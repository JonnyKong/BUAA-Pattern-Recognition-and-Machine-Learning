function model = train_nn(train_x, train_y, test_x, test_y, neuron_num, learning_rate, epoch_num, learning_rate_decay, lambda, maxX, minX, meanX, stdX)
% Whether in autoencoder mode
autoencoder = false;
if(nargin >= 10)
    autoencoder = true;
end
autoencoder_input = [];
autoencoder_output = [];

% Ensure neuron number match input data
assert(size(train_x, 2) == neuron_num(1));
assert(size(train_y, 2) == neuron_num(end));

% Use softmax in multivariate conditions
softmax = false;
if(neuron_num(end) > 1 && autoencoder == false)
    softmax = true;
end

% Construct neural Network
layer_num = length(neuron_num);
% Construct input and activation vectors
input = cell(1, layer_num);
activation = cell(1, layer_num);
for i = 1 : layer_num
    input{i} = zeros(neuron_num(i), 1);
    activation{i} = zeros(neuron_num(i), 1);
end
% Construct the weight matrices(including a bias vector as the last column)
weight = cell(1, layer_num - 1);
for i = 1 : layer_num - 1
    % Initialize the weight to be a random number in [-e, e]
    % where e is determined by the number of input and output neurons
    epsilon_init = sqrt(6) / sqrt(neuron_num(i) + neuron_num(i + 1));
    weight{i} = rand(neuron_num(i + 1), neuron_num(i) + 1) * 2 * epsilon_init - epsilon_init;
end
% Construct the delta vectors(with the first element being empty)
delta = cell(1, layer_num);
for i = 2 : layer_num
    delta{i} = zeros(neuron_num(i), 1);
end


% Train Neural Network
fprintf('Training...\n')
best_testset_cost = 100;
best_accuracy = 0;
num_not_improving = 0;
train_cost = [];
test_cost = [];
for epoch = 1 : epoch_num
    autoencoder_input = [];
    autoencoder_output = [];
    fprintf('Iteration %d\n', epoch);
    % Initialize the accumulative derivative matrix corresponding to each
    % weight matrix
    accu = cell(1, layer_num - 1);
    for i = 1 : layer_num - 1
        accu{i} = zeros(neuron_num(i + 1), neuron_num(i) + 1);
    end
    
    % Traverse the train dataset
    num_correct = 0;
    J = 0;
    for i = 1 : size(train_x, 1)
        % Forward Propagation
        input{1} = train_x(i, :)';
        activation{1} = input{1};
        for j = 2 : layer_num
            input{j} = weight{j - 1} * [activation{j - 1}; 1];
            activation{j} = sigmoid(input{j});
        end
        % Softmax
        if(softmax)
            sum_of_probs = sum(activation{layer_num});
            activation{layer_num} = activation{layer_num} / sum_of_probs;
        end
        % Back Propagation
        delta{layer_num} = activation{layer_num} - [train_y(i, :)]';
        for j = layer_num - 1 : 2
            % Compute the delta terms on previous layers
            delta{j} = (weight{j}(:, 1 : end - 1))' * delta{j + 1} .* (activation{j} .* (1 - activation{j}));
        end
        % Accumulate the delta terms on corresponding layers
        for j = 1 : layer_num - 1
            accu{j} = accu{j} + delta{j + 1} * [activation{j}; 1]';
        end
        % Accumulate the cost for this sample
        J = J - (train_y(i, :) * log(activation{layer_num}) + (1 - train_y(i, :)) * log(1 - activation{layer_num}));
        % Compute accuracy
        if(softmax)
            [~, maxPos] = max(activation{layer_num});
            activation{layer_num} = zeros(size(activation{layer_num}));
            activation{layer_num}(maxPos) = 1;
        else
            activation{layer_num}(activation{layer_num} > 0.5) = 1;
            activation{layer_num}(activation{layer_num} <= 0.5) = 0;
        end
        if(autoencoder == false)
            if(isequal(activation{layer_num}, train_y(i, :)'))
                num_correct = num_correct + 1;
            end
        end
    end
    J = J / size(train_x, 1);
    % Calculate Weight Decay
    for i = 1 : layer_num - 1
        J = J + (lambda / 2) * sum(sum(weight{i} .* weight{i}));
    end
    train_cost = [train_cost, J];
    fprintf('Training Set Cost: %f\n', J);
    if(autoencoder == false)
        num_correct = num_correct / size(train_x, 1);
        fprintf('Training Set Accuracy: %f\n', num_correct)
    end
    
    % Update the weights
    for i = 1 : layer_num - 1
        weight{i} = weight{i} - learning_rate * ((1 / size(train_x, 1)) * accu{i} + lambda * [weight{i}(:, 1 : end - 1), zeros(size(weight{i}, 1), 1)]);
    end
    
    % Traverse the test dataset
    num_correct = 0;
    J = 0;
    true_positive = 0;
    false_positive = 0;
    false_negative = 0;
    true_negative = 0;
    for i = 1 : size(test_x, 1)
        % Forward propagation
        input{1} = test_x(i, :)';
        activation{1} = input{1};
        if(autoencoder)
            x_actual = test_x(i, :) .* (stdX) + meanX;
            if(isempty(autoencoder_input))
                autoencoder_input = x_actual;
            else
                autoencoder_input = [autoencoder_input; x_actual];
            end
        end
        for j = 2 : layer_num
            input{j} = weight{j - 1} * [activation{j - 1}; 1];
            activation{j} = sigmoid(input{j});
        end
        if(autoencoder)
            x_predict = (activation{layer_num})' .* (maxX - minX) + minX;
            if(isempty(autoencoder_output))
                autoencoder_output = x_predict;
            else
                autoencoder_output = [autoencoder_output; x_predict];
            end
        end
        % Last Layer is Softmax
        if(softmax)
            sum_of_probs = sum(activation{layer_num});
            activation{layer_num} = activation{layer_num} / sum_of_probs;
        end
        % Accumulate the cost for this sample
        J = J - (test_y(i, :) * log(activation{layer_num}) + (1 - test_y(i, :)) * log(1 - activation{layer_num}));
        % Compute accuracy
        if(softmax)
            [~, maxPos] = max(activation{layer_num});
            activation{layer_num} = zeros(size(activation{layer_num}));
            activation{layer_num}(maxPos) = 1;
        else
            activation{layer_num}(activation{layer_num} > 0.5) = 1;
            activation{layer_num}(activation{layer_num} <= 0.5) = 0;
        end
        if(autoencoder == false)
            if(isequal(activation{layer_num}, test_y(i, :)'))
                num_correct = num_correct + 1;
            end
        end
        % Compute F1 score for two-class classification cases
        if(~softmax && ~autoencoder)
            if(activation{layer_num} == 1 && test_y(i, :) == 1)
                true_positive = true_positive + 1;
            elseif(activation{layer_num} == 1 && test_y(i, :) == 0)
                false_positive = false_positive + 1;
            elseif(activation{layer_num} == 0 && test_y(i, :) == 1)
                false_negative = false_negative + 1;
            else
                true_negative = true_negative + 1;
            end
        end
    end
    J = J / size(test_x, 1);
    % Calculate Weight Decay
    for i = 1 : layer_num - 1
        J = J + (lambda / 2) * sum(sum(weight{i} .* weight{i}));
    end
    test_cost = [test_cost, J];
    precision = true_positive / (true_positive + false_positive);
    recall = true_positive / (true_positive + false_negative);
    F1_score = (2 * precision * recall) / (precision + recall);
    fprintf('Test Set Cost: %f\n', J);
    if(autoencoder == false)
        num_correct = num_correct / size(test_x, 1);
        fprintf('Test Set Accuracy: %f\n', num_correct);
    end
    if(~softmax && ~autoencoder)
        fprintf('Test Set F1 Score: %f\n\n', F1_score);
    else
        fprintf('\n')
    end
    if(mod(epoch, 100) == 0)
        learning_rate = learning_rate * learning_rate_decay;
    end
    % Record the best test data performance
    if(J < best_testset_cost)
        best_testset_cost = J;
        num_not_improving = 0;
        if(autoencoder == false)
            best_accuracy = num_correct;
        end
    % Early Stopping
    else
        num_not_improving = num_not_improving + 1;
        if(num_not_improving >= 10000)
            fprintf('Early Stopped at epoch %d\n', epoch);
            break
        end
    end
end

% Display results
if(autoencoder == false)
    fprintf('Best Test Accuracy: %f\n', best_accuracy);
end

% Plot learning curves
figure(1)
hold on
x = 1 : length(train_cost);
plot(x, train_cost, '--b', 'LineWidth', 2);
plot(x, test_cost, '-.g', 'LineWidth', 2);
legend('Train', 'Test')
% plot(x, test_cost, '-.g', 'LineWidth', 2);
% legend(num2str(learning_rate))

% Print autoencoder to excel
% headers = ['a1', 'a2', 'a3', 'a4'];
% xlswrite('input.xlsx', autoencoder_input)
% xlswrite('output.xlsx', autoencoder_output)

% Save neural network
model.weight = weight;
end