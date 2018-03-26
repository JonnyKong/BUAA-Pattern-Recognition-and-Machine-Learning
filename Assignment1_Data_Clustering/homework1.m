%% Generate Dataset
mean1_init = [-2 -3.5];
cov1_init = [0.5 -0.3; -0.3 1];
mean2_init = [-1 4];
cov2_init = [1 0; 0 0.5];
mean3_init = [1.5, -1];
cov3_init = [0.7 0; 0 0.7];
num_of_samples_per_class = 30;
w1 = mvnrnd(mean1_init, cov1_init, num_of_samples_per_class);
w2 = mvnrnd(mean2_init, cov2_init, num_of_samples_per_class);
w3 = mvnrnd(mean3_init, cov3_init, num_of_samples_per_class);
w = [w1; w2; w3];

%% clustering
% label = k_means(w);
label = dbscan(w, 8, 1.6);
% Plot original Data
title('Result')
figure(1)
hold on
% plot(w1(:, 1), w1(:, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
% plot(w2(:, 1), w2(:, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7)
% plot(w3(:, 1), w3(:, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
plot(w(:, 1), w(:, 2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7)
title('Original')
figure(2)
hold on
plot(w(label == 1, 1), w(label == 1, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
plot(w(label == 2, 1), w(label == 2, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
plot(w(label == 3, 1), w(label == 3, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);


%% Functions
function label = k_means(w)
minPos = min(w);
maxPos = max(w);
% Init centroids and normalize 
centroids = rand(3, size(w, 2));
centroids(:, 1) = centroids(:, 1) * (maxPos(1) - minPos(1)) + minPos(1);
centroids(:, 2) = centroids(:, 2) * (maxPos(2) - minPos(2)) + minPos(2);
figure(1)
hold on
label = zeros(size(w, 1), 1);
while(1)
    label_prev = label;
    label = findNearest(w, centroids);
    % Plot
    plt1 = plot(w(label == 1, 1), w(label == 1, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    plt2 = plot(w(label == 2, 1), w(label == 2, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
    plt3 = plot(w(label == 3, 1), w(label == 3, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
    plt4 = plot(centroids(1, 1), centroids(1, 2), 'y+', 'LineWidth', 1, 'MarkerSize', 10);
    plt5 = plot(centroids(2, 1), centroids(2, 2), 'b+', 'LineWidth', 1, 'MarkerSize', 10);
    plt6 = plot(centroids(3, 1), centroids(3, 2), 'r+', 'LineWidth', 1, 'MarkerSize', 10);
    pause(0.5);
    % Assign new pos to centroids
    centroids = updatePosition(w, label, centroids);
    % Check convergence
    if(sum(label == label_prev) == size(w, 1))
        disp('Converged')
        break;
    end
    delete(plt1) 
    delete(plt2)
    delete(plt3)
    delete(plt4)
    delete(plt5)
    delete(plt6)
end
end

function label = dbscan(w, minPts, epsilon)
import java.util.LinkedList
% Parameters
distance = calculateDistance(w);
class_cnt = 1;
visited = zeros(size(w, 1), 1);
isCentroid = zeros(size(w, 1), 1);
label = zeros(size(w, 1), 1);
% Determine centroids
for i = 1 : size(w, 1)
    D = distance(i, :);
    neighbor = find(D <= epsilon);
    if(length(neighbor) >= minPts)
        isCentroid(i) = 1;
    end
end
while(sum(visited == 0) > 0)
    visited_prev = visited;
    % Find a random centroid
    randomCentroidPos = randi(size(w, 1));
    while(isCentroid(randomCentroidPos) == 0)
        randomCentroidPos = randomCentroidPos + 1;
        if(randomCentroidPos > size(w, 1))
            randomCentroidPos = 1;
        end
    end
    % Initialize queue
    Q = LinkedList();
    Q.add(randomCentroidPos);
    visited(randomCentroidPos) = 1;
    while(Q.size() > 0)
        q = Q.pop();
        if(isCentroid(q) == 1)
            for i = 1 : size(w, 1)
                if(distance(q, i) <= epsilon && visited(i) == 0 && i ~= q)
                    visited(i) = 1;
                    Q.add(i);
                end
            end
        end
    end
    label(visited ~= visited_prev) = class_cnt;
    isCentroid(visited ~= visited_prev) = 0;
    class_cnt = class_cnt + 1;
end
end

function label = findNearest(w, centroids)
label = zeros(size(w, 1), 1);
for i = 1 : size(w, 1)
    minDistance = 1e+5;
    minDistCentroid = 1;
    for j = 1 : size(centroids, 1)
        distance = 0;
        for k = 1 : size(w, 2)
            distance = distance + (w(i, k) - centroids(j, k)) ^ 2;
        end
        if(distance < minDistance || j == 1)
            minDistance = distance;
            minDistCentroid = j;
        end
    end
    label(i) = minDistCentroid;
end
end

function distance = calculateDistance(w)
distance = zeros(size(w, 1), size(w, 1));
for i_ = 1 : size(w, 1)
    for j_ = i_ : size(w, 1)
        tmp = 0;
        for k = 1 : size(w, 2)
            tmp = tmp + (w(i_, k) - w(j_, k)) ^ 2;
        end
        tmp = sqrt(tmp);
        distance(i_, j_) = tmp;
        distance(j_, i_) = tmp;
    end
end
end

function centroids_new = updatePosition(w, label, centroids)
centroids_new = zeros(size(centroids));
for i = 1 : size(centroids, 1)
    centroids_new(i, :) = mean(w(label == i, :));
end
end