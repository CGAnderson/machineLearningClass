function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
[m,n] = size(X); % m is number of rows (examples), n is number of columns (features)

 mu = mean(X) %computes the mean of each column in X
 sigma = std(X)

 for i=1:n
     X_norm(:,i) = (X(:,i) - mu(1,i))/sigma(1,i);
 end
 
fprintf('X_norm = (X - mu(1,:))/sigma(1,:) ...\n');
C = bsxfun(@minus,X,mu); % also suggested to use, C = A - B(ones(size(A,1),1),:);  or C = A-diag(B)*ones(size(A));
                % size(A,1) gives number of rows of A
                %ones(rows,1) gives ones array of rows by 1 column
                % same as multiplying mu by ones(rows,1) and giving a new
                % matrix of (A number of row and 1 col)times (1 row and mu number of colums)
                %same as ones(size(A,1),1)*mu which is mean of each col of
                %A, therefore is a 1 row, n col vector
fprintf('mu(ones(size(X,1),1),:)...\n');
size(X,1)
ones(size(X,1),1)
mu
valOfmu = mu(ones(size(X,1),1),:)   


X_norm = bsxfun(@rdivide, C, sigma); %divide by right array C./sigma

% ============================================================

end
