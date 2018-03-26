%% 
w = generate_spiral();
plot_dataset_helper(w);

%% 
function plot_dataset_helper(w)
figure(1)
plot(w(:, 1), w(:, 2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7)
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