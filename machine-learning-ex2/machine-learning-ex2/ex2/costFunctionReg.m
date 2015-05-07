function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = sigmoid( X*theta); %predictions of hypothesis on m..X*theta gives mby1 matrix
%firstTerm = -y' * (log(predictions)); %works either way with or without
%transpose...the error was with the lambda stuff, I think
%secondTerm = (1 - y') * (log(1 - predictions));
firstTerm = -y .* (log(predictions));
secondTerm = (1 - y) .* (log(1 - predictions));
J = (1/m) * sum(firstTerm - secondTerm) ; %+ lambda * sum(theta(2:end).^2) / (2*m);;

%theta
%lambda
%[r,c] = size(theta);
%thetaSub = [0; theta(2:end,:)];
thetaSub = theta(2:end);
%Jsub = (lambda/ (2*m))*sum(theta(2:end) .^2); % not putting () around 2m changed the result!!!
%J = J + Jsub;
%thetaSub = theta(2:r,:);
Jsub = (lambda/(2*m))*sum(thetaSub .* thetaSub); %both work now, it was the () issue not it's placement before/after
J = J + Jsub;

thetaSub = [0; theta(2:end)];
diffTerm = (predictions - y);
%note: below way with bsxfun and sum(diff) results in same solution as
%other way where diff' * X is used instead
%diffTerm3a = bsxfun(@times,diffTerm, X);
%gradBSXway = (1/m)* sum(diffTerm3a)
diffTerm3 = diffTerm' * X; % bsxfun(@times,diffTerm, X); if use bsxFun then also need sum(diffTerm3) in next line
grad = (1/m) * ( diffTerm3 );  %note about sum, doc's say  sum(A) returns the sum of the elements of A along the first array dimension whose size does not equal 1.
                            %so when grad was mxn (ie bsxfun element wise
                            %multiplication....then sum(diffterm) is sum
                            %along each row? actually means since row not 1
                            %sum will sum each column, and put result in
                            %each col location of resulting row vector
%this one works
%grad =((predictions - y)' * X / m)';

gradSub = (lambda/m)*(thetaSub) 
grad = grad + gradSub'; %above, my orig method, works if I transpose grad here...other sol'n grad isn't transposed here
%note, I can also transpose gradSub and it's ok (now a col vector is
%all)..where grad' will give a row vector

%the real issue seems to be with sum(thetaSub)...NO! that is a scalar, not a vector result,
% and other issue previously with  () with J above and 2m

%[r,c] = size(grad); %one row, 28 col
%subset = grad(2:r,:) 
%subset = subset + gradSub


%hx = sigmoid(X * theta);
%m = length(X);

%J = (sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m) + lambda * sum(theta(2:end).^2) / (2*m);
%grad =((hx - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;



% =============================================================

end
