function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    delta = (1/m)*sum(X.*repmat((X*theta - y), 1, size(X,2))); % 看不懂这个
    theta = (theta' - (alpha * delta))';

    % this works too:
    %    h = X*theta; % m*(n+1) (n+1)*1 -> m*1  , in this case, n == 1
    %    diff = h-y; % m*1  
    %    theta_change = (X'*diff)*alpha/m; % (n+1)*m × m*1 , (n+1)*1 vectorization
    %    theta = theta - theta_change;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
