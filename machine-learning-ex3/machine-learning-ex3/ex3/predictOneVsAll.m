function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
all_preds = all_theta * X';  %gives 10 by 5000 matrix (10by401 alltheta times 5000x401 X..1 added)
maxValEachrow = max(all_preds,[],2); %give me a col vector with max from each row
size(maxValEachrow)
maxValEachcol = max(all_preds,[],1); %means give me a row with the maximum from each col
size(maxValEachcol)
%max(all_preds) %max value for each col
%then [max, i] will give i=index of that max, 
%i.e. which row did the max for that column come from?
[max_vals, i] = max(all_preds,[],1); %gives two row vectors...i.e. items in columns (horizontal)
%row 10, col 111 7.6791 
%value10111 = all_preds(10,111)
%p = i; %p is left as row vector, but this is not how it will be used..see above in main info, it shows p being used like a col vector ..i.e. the ; on each element

p = i'; %p is transposed, so now a col vector (items in rows...vertical)
%size(p)




% =========================================================================


end
