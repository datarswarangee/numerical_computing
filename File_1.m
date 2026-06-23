clc; clear; close all;

data = load('assignmentSegmentBrain.mat');

Y = double(data.imageData);
mask = double(data.imageMask);

figure; imshow(Y,[]); title('Input MR Image');
figure; imshow(mask,[]); title('Brain Mask');

%% Parameters
K = 3;                 % Number of classes
q = 2.0;               % Fuzziness parameter (>1.5)
maxIter = 50;
epsilon = 1e-5;

%% Get brain pixels
idx = find(mask > 0);
Yb = Y(idx);

N = length(Yb);

%% Initialize Bias Field (constant)
b = ones(size(Y));

%% Initialize Memberships (random but normalized)
U = rand(N, K);
U = U ./ sum(U,2);

%% Initialize Class Means (using percentiles)
c = quantile(Yb, [0.2 0.5 0.8]);

%% Neighborhood Mask (Gaussian)
win_size = 5;
sigma = 1.0;

[x,y] = meshgrid(-(win_size-1)/2:(win_size-1)/2);
w = exp(-(x.^2 + y.^2)/(2*sigma^2));
w = w / sum(w(:));

figure; imshow(imresize(w,20,'nearest'),[]);
title('Neighborhood Mask');

%% Objective function storage
obj_vals = [];

%% Iterations
for iter = 1:maxIter
    
    %% --- Step (a): Update Class Means ck ---
    for k = 1:K
        num = 0;
        den = 0;
        
        for n = 1:N
            [r,c_idx] = ind2sub(size(Y), idx(n));
            
            % Extract neighborhood
            for i = -2:2
                for j = -2:2
                    rr = r+i;
                    cc = c_idx+j;
                    
                    if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                        w_ij = w(i+3,j+3);
                        num = num + w_ij * (U(n,k)^q) * Y(rr,cc) * b(rr,cc);
                        den = den + w_ij * (U(n,k)^q) * (b(rr,cc)^2);
                    end
                end
            end
        end
        
        c(k) = num / (den + eps);
    end

    %% --- Step (b): Update Memberships ---
    for n = 1:N
        [r,c_idx] = ind2sub(size(Y), idx(n));
        
        d = zeros(1,K);
        
        for k = 1:K
            temp = 0;
            
            for i = -2:2
                for j = -2:2
                    rr = r+i;
                    cc = c_idx+j;
                    
                    if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                        w_ij = w(i+3,j+3);
                        temp = temp + w_ij * (Y(rr,cc) - c(k)*b(rr,cc))^2;
                    end
                end
            end
            
            d(k) = temp + eps;
        end
        
        for k = 1:K
            denom = 0;
            for j = 1:K
                denom = denom + (d(k)/d(j))^(1/(q-1));
            end
            U(n,k) = 1/denom;
        end
    end

    %% --- Step (c): Update Bias Field ---
    for n = 1:N
        [r,c_idx] = ind2sub(size(Y), idx(n));
        
        num = 0;
        den = 0;
        
        for i = -2:2
            for j = -2:2
                rr = r+i;
                cc = c_idx+j;
                
                if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                    w_ij = w(i+3,j+3);
                    
                    for k = 1:K
                        num = num + w_ij * (U(n,k)^q) * Y(rr,cc) * c(k);
                        den = den + w_ij * (U(n,k)^q) * (c(k)^2);
                    end
                end
            end
        end
        
        b(r,c_idx) = num / (den + eps);
    end

    %% --- Objective Function ---
    J = 0;
    for n = 1:N
        [r,c_idx] = ind2sub(size(Y), idx(n));
        
        for k = 1:K
            temp = 0;
            
            for i = -2:2
                for j = -2:2
                    rr = r+i;
                    cc = c_idx+j;
                    
                    if rr>0 && cc>0 && rr<=size(Y,1) && cc<=size(Y,2)
                        w_ij = w(i+3,j+3);
                        temp = temp + w_ij * (Y(rr,cc) - c(k)*b(rr,cc))^2;
                    end
                end
            end
            
            J = J + (U(n,k)^q) * temp;
        end
    end
    
    obj_vals = [obj_vals J];
    
    % Convergence check
    if iter > 1 && abs(obj_vals(end) - obj_vals(end-1)) < epsilon
        break;
    end
    
    fprintf('Iteration %d, Objective = %f\n', iter, J);
end

%% Construct Bias Removed Image A
A = zeros(size(Y));
for n = 1:N
    [r,c_idx] = ind2sub(size(Y), idx(n));
    A(r,c_idx) = sum(U(n,:) .* c);
end

%% Residual Image
R = Y - A .* b;

%% Membership Images
U_img = zeros([size(Y), K]);
for k = 1:K
    temp = zeros(size(Y));
    temp(idx) = U(:,k);
    U_img(:,:,k) = temp;
end

%% Display Results
figure;
subplot(2,3,1); imshow(Y,[]); title('Corrupted Image');
subplot(2,3,2); imshow(U_img(:,:,1),[]); title('Class 1');
subplot(2,3,3); imshow(U_img(:,:,2),[]); title('Class 2');
subplot(2,3,4); imshow(U_img(:,:,3),[]); title('Class 3');
subplot(2,3,5); imshow(b,[]); title('Bias Field');
subplot(2,3,6); imshow(A,[]); title('Bias Removed');

figure; imshow(R,[]); title('Residual Image');

figure; plot(obj_vals,'LineWidth',2);
title('Objective Function vs Iteration');
xlabel('Iteration'); ylabel('J');
