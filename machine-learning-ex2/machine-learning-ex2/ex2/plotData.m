function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%create set of two indices values of y, 1 and 0...which we will
%use to reference which X (m by 2) data we want for each indice set
posYrow = find(y==1);
negYrow = find(y==0);

%plot each set on same plot but with different markers
%need to plot x,y...but here x, y are the two test scores
%so need X col 1 and X col 2...and particular rows (that correspond to y=1,
%y=0
plot(X(posYrow,1), X(posYrow,2), 'k+', 'LineWidth', 1, 'MarkerEdgeColor','b','MarkerFaceColor', 'g', 'MarkerSize', 6);

plot(X(negYrow,1), X(negYrow,2), 'ko', 'LineWidth', 1, 'MarkerEdgeColor','k','MarkerFaceColor', 'r', 'MarkerSize', 6);







% =========================================================================



hold off;

end
