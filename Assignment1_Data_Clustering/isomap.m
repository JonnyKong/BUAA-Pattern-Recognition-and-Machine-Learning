w = generate_spiral();
figure(1)
label = isomapClustering(w, 4);
plot(w(label == 1, 1), w(label == 1, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold on
plot(w(label == 2, 1), w(label == 2, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7)
title('Isomap Clustering')

%% functions
function label = isomapClustering(w, epsilon)
distance = calculateDistance(w, epsilon);
distance = calculateManifoldDistance(distance);
X = mds(distance, 1);
label = zeros(size(w, 1), 1);
label(X >= 0) = 1;
label(X < 0) = 2;
end

function X = mds(distance, target_dimension)
B = zeros(size(distance));
n = size(B, 1);
for i__ = 1 : size(distance, 1)
    for j__ = 1 : size(distance, 1)
        B(i__, j__) = -0.5 * (distance(i__, j__) ^ 2 - distance(i__, :) * distance(i__, :)' / n - distance(:, j__)' * distance(:, j__) / n + sum(sum(distance .^ 2)) / (n ^ 2));
    end
end
[V, D] = eig(B);
X = V(:, 1 : target_dimension) * D(1 : target_dimension, 1 : target_dimension) .^ (1 / 2);
end

% Calculate the distance of k-nearest-neighbors, otherwise infinite
function distance = calculateDistance(w, epsilon)
distance = zeros(size(w, 1), size(w, 1));
for i_ = 1 : size(w, 1)
    for j_ = i_ : size(w, 1)
        tmp = 0;
        for k = 1 : size(w, 2)
            tmp = tmp + (w(i_, k) - w(j_, k)) ^ 2;
        end
        tmp = sqrt(tmp);
        if(tmp < epsilon)
            distance(i_, j_) = tmp;
            distance(j_, i_) = tmp;
        else
            distance(i_, j_) = inf;
            distance(j_, i_) = inf;
        end
    end
end
end

% Calculate shortest path between paths
function manifoldDistance = calculateManifoldDistance(distance)
manifoldDistance = zeros(size(distance));
num_samples = size(distance, 1);
distance(isinf(distance)) = 0;
distance = sparse(distance);
for i_ = 1 : num_samples
    [d, ~, ~] = graphshortestpath(distance, i_);
    manifoldDistance(i_, :) = d;
end
end

function w = generate_spiral()
r = 2;
theta = 0;
w1 = [];
w2 = [];
for i = 1 : 30
    x = [r * cos(theta * pi / 180), r * sin(theta * pi / 180)];
    w1 = [w1; x];
    w2 = [w2; -x];
    r = r + 0.5;
    theta = theta + 80 / r;
end
w = [w1; w2];
end

function w = generate_concentric()
w = [];
i = 0;
while(i < 200)
    x = rand() * 20 - 10;
    y = rand() * 20 - 10;
    if(x ^ 2 + y ^ 2 <= 10)
        w = [w; [x y]];
    elseif(x ^ 2 + y ^ 2 >= 50 && x ^ 2 + y ^ 2 <= 100)
        w = [w; [x y]];
    else
        continue;
    end
    i = i + 1;
end
end