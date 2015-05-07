function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for i = 1:m
    temp = ((X(i,:)*theta - y(i,:))^2);
    J = J + temp;
end
   % myJ = (2*m)^(-1)*J
    J = (1/(2*m))*J;
    
%his way ...equivalent but  I need to understand why
m1 = size(X,1); %number of examples
%if m == m1, disp('m and m1 are the same at: '), m, m1;
predictions = X*theta; %predictions of hypothesis on m
sqrErrors = (predictions - y) .^2;
J = 1/(2*m1) * sum(sqrErrors);

stdErrorOfEst = (sum(sqrErrors)/m1) .^(1/2);
meanPriceSample = mean(y);
relstd = (stdErrorOfEst/meanPriceSample)*100;
fprintf(['J is :\n $%f\n and stdError of the regression estimate' ...
    '\n sqroot of sum of (prediction - actual)squared' ...
    'then div by total \n for this cost J is : $%f\n  and mean of actual prices ' ...
    'is: $%f\n and relative std error of estimate i.e. std err of est/mean is: %f%%\n\n'], ...
    J, stdErrorOfEst, meanPriceSample, relstd);




% =========================================================================

end
