function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% g(z) = 1/(1 + e to the negative z)  in matlab e to z is exp(z)

g = (1 + exp(-z)) .^(-1);



% =============================================================

end
