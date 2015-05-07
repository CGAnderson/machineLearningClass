function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%first, the 3 layer neural network using theta1 and 2 weights
%calculate h of theta x over K range:
%start with input layer
a1 = X'; %transpose now, so can multiply with Theta1
%add a0at1
a1 = [ones(1,size(a1,2)); a1];
%compute hidden layer
%size(a1)
%size(Theta1)
z2 = Theta1*a1;
a2 = sigmoid(z2); %i.e. g(z2)
%add a0at2
a2 = [ones(1,size(a2,2)); a2];
%compute output layer
z3 = Theta2*a2;
a3 = sigmoid(z3); %i.e. g(z3) which equals h theta(x) over K, the hypothesis
hOverK = a3;
%size(hOverK) %10 by 5000
%size(y) %5000 by 1

%next, the cost function, J without lambda
%%choose the hypothesis max over the K range (i.e. h(x)overK
%[max_hypothesisVal, index_of_max] = max(a3);
%hOverK = index_of_max'; %as col vector because it's used that way later ex3_nn

%right now, y is a 5000by1 list of labels (10 to 9, with 10=0)
%turn y into a vector where a 1 is in place of the ith element, where
%i=label number
yAsVector = zeros(size(hOverK,1), size(hOverK,2));
%size(yAsVector)
for idx = 1:m,
   yAsVector(y(idx), idx) = 1; %put a one in the spot for that ith index/label value 
end
predictions = hOverK; %predictions of hypothesis on m..
%size(predictions)
firstTerm = -yAsVector .* (log(predictions));
secondTerm = (1 - yAsVector) .* (log(1 - predictions));
JVectorValues = (1/m) * sum(firstTerm - secondTerm); 

J = sum(JVectorValues);

%add in the regularization, lambda
%this time, we have two theta to worry about
%remove the bias from each, the first column from each theta matrix
thetaSub1 = Theta1(:,2:end);
thetaSub2 = Theta2(:,2:end);
Jsub = (lambda/(2*m))* (sum(sum(thetaSub1 .* thetaSub1)) + sum(sum(thetaSub2 .* thetaSub2))); 
J = J + Jsub;


d3 = a3 - yAsVector;
%d2 = (Theta2' * d3)*  sigmoidGradient(z2)' ;
%d2sub = d2(:,2:end);

grad1_accumulate = 0;
grad2_accumulate = 0;
for idx = 1:m
   d2 = Theta2' * d3(:,idx);
%size(sigmoidGradient(z2(:,idx)))
%size(d2)
   d2 = d2(2:end) .* sigmoidGradient(z2(:,idx));
   grad1_accumulate = grad1_accumulate + d2*a1(:,idx)';
   grad2_accumulate = grad2_accumulate + d3(:,idx)*a2(:,idx)';
end

Theta1_grad = (1/m)*grad1_accumulate;
Theta2_grad = (1/m)*grad2_accumulate;

rVal1 = (lambda/m)*Theta1(:,2:end);
rVal2 = (lambda/m)*Theta2(:, 2:end);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + rVal1;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + rVal2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
