function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%me, start with input layer
a1 = X'; %transpose now, so can multiply with Theta1
%add a0at1
a1 = [ones(1,size(a1,2)); a1];
%compute hidden layer

size(a1)
size(Theta1)
z2 = Theta1*a1;
a2 = sigmoid(z2); %i.e. g(z2)
%add a0at2
a2 = [ones(1,size(a2,2)); a2];
%compute output layer
z3 = Theta2*a2;
a3 = sigmoid(z3); %i.e. g(z3) which equals h0(x), the hypothesis

%choose the hypothesis max over the K range (i.e. h(x)overK
[max_hypothesisVal, index_of_max] = max(a3);
p = index_of_max'; %as col vector because it's used that way later ex3_nn








% =========================================================================


end
