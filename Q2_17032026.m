clc; clear; close all;

%% Load Data
data = load('assignmentSegmentBrainGmmEmMrf.mat');

Y = double(data.imageData);
mask = double(data.imageMask);

figure; imshow(Y,[]); title('Corrupted Image');

%% Parameters
K = 3;
beta = 0.8;
maxIter = 30;
eps_val = 1e-6;

[idx_r, idx_c] = find(mask > 0);
N = length(idx_r);

Yb = Y(mask > 0);
Yb = Yb(:);   % सुनिश्चित column vector

%% ---- Initialization ----

[init_labels, ~] = kmeans(Yb, K);

labels = zeros(size(Y));
labels(mask>0) = init_labels;

mu = zeros(1,K);
sigma = zeros(1,K);

for k = 1:K
    pixels = Yb(init_labels == k);
    
    if isempty(pixels)
        mu(k) = mean(Yb);
        sigma(k) = std(Yb) + eps;
    else
        mu(k) = mean(pixels);
        sigma(k) = std(pixels) + eps;
    end
end

%% Memberships
U = zeros(N,K);

%% Neighborhood (4-connectivity)
nbr = [0 1; 1 0; 0 -1; -1 0];

log_post = [];

%% EM Iterations
for iter = 1:maxIter
    
    %% -------- (a) E-step --------
    for n = 1:N
        r = idx_r(n);
        c = idx_c(n);
        
        probs = zeros(1,K);
        
        for k = 1:K
            
            % Gaussian likelihood (stable)
            diff = Y(r,c) - mu(k);
            p_y = exp(- (diff^2) / (2*sigma(k)^2)) / (sqrt(2*pi)*sigma(k) + eps_val);
            
            % MRF prior
            mrf_energy = 0;
            for t = 1:4
                rr = r + nbr(t,1);
                cc = c + nbr(t,2);
                
                if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                    if mask(rr,cc) > 0
                        if labels(rr,cc) ~= k
                            mrf_energy = mrf_energy + 1;
                        end
                    end
                end
            end
            
            prior = exp(-beta * mrf_energy);
            probs(k) = p_y * prior;
        end
        
        probs = probs + eps_val;
        U(n,:) = probs ./ sum(probs);
    end

    %% -------- (b) Update Means --------
    for k = 1:K
        w = U(:,k);
        num = sum(w .* Yb);
        den = sum(w);
        
        if den < eps_val
            mu(k) = mean(Yb);
        else
            mu(k) = num / den;
        end
    end

    %% -------- (c) Update Std --------
    for k = 1:K
        w = U(:,k);
        num = sum(w .* (Yb - mu(k)).^2);
        den = sum(w);
        
        if den < eps_val
            sigma(k) = std(Yb) + eps;
        else
            sigma(k) = sqrt(num / den + eps_val);
        end
    end

    %% -------- (d) ICM Label Update --------
    
    logP_before = compute_log_posterior(Y, labels, mu, sigma, beta, mask);
    
    new_labels = labels;
    
    for n = 1:N
        r = idx_r(n);
        c = idx_c(n);
        
        best_label = labels(r,c);
        best_energy = inf;
        
        for k = 1:K
            
            % Data energy
            diff = Y(r,c) - mu(k);
            E_data = (diff^2)/(2*sigma(k)^2) + log(sigma(k) + eps_val);
            
            % MRF energy
            E_mrf = 0;
            for t = 1:4
                rr = r + nbr(t,1);
                cc = c + nbr(t,2);
                
                if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                    if mask(rr,cc) > 0
                        if new_labels(rr,cc) ~= k
                            E_mrf = E_mrf + 1;
                        end
                    end
                end
            end
            
            total_energy = E_data + beta * E_mrf;
            
            if total_energy < best_energy
                best_energy = total_energy;
                best_label = k;
            end
        end
        
        new_labels(r,c) = best_label;
    end
    
    labels = new_labels;
    
    logP_after = compute_log_posterior(Y, labels, mu, sigma, beta, mask);
    
    log_post = [log_post; logP_before logP_after];
    
    fprintf('Iter %d: Before = %f | After = %f\n', ...
        iter, logP_before, logP_after);
end

%% Membership Images
U_img = zeros([size(Y), K]);

for k = 1:K
    temp = zeros(size(Y));
    temp(mask>0) = U(:,k);
    U_img(:,:,k) = temp;
end

%% Display
figure;
subplot(2,3,1); imshow(Y,[]); title('Corrupted');
subplot(2,3,2); imshow(U_img(:,:,1),[]); title('Class 1');
subplot(2,3,3); imshow(U_img(:,:,2),[]); title('Class 2');
subplot(2,3,4); imshow(U_img(:,:,3),[]); title('Class 3');
subplot(2,3,5); imshow(labels,[]); title('Final Labels');

figure;
plot(log_post);
legend('Before ICM','After ICM');
title('Log Posterior');

%% -------- Helper Function --------
function logP = compute_log_posterior(Y, labels, mu, sigma, beta, mask)

[idx_r, idx_c] = find(mask > 0);
nbr = [0 1; 1 0; 0 -1; -1 0];

logP = 0;

for n = 1:length(idx_r)
    r = idx_r(n);
    c = idx_c(n);
    k = labels(r,c);
    
    diff = Y(r,c) - mu(k);
    logP = logP - (diff^2)/(2*sigma(k)^2) - log(sigma(k) + eps);
    
    for t = 1:4
        rr = r + nbr(t,1);
        cc = c + nbr(t,2);
        
        if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
            if mask(rr,cc) > 0
                if labels(rr,cc) ~= k
                    logP = logP - beta;
                end
            end
        end
    end
end

end