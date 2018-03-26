%% Generate dataset
% w = generate_concentric();
% label = dbscan(w, 5, 2);
w = generate_spiral();
label = dbscan(w, 2, 2);
figure(1)
plot(w(label == 1, 1), w(label == 1, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold on
plot(w(label == 2, 1), w(label == 2, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7)
title('DBSCAN Clustering')

%% DBSCAN
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

%% Generate distance matrix of the samples
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


