% @Author: Athul Vijayan
% @Date:   2014-05-23 12:01:18
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-05-23 22:37:20

%% logreg: Function does a logistic regression with given training set and returns parameters after learning
function [theta] = logreg(inputs, outputs)
	inputs = [ones(size(inputs, 1), 1) inputs];
	theta = rand(size(inputs, 2), 1);

	max_iter = 50;     % maximum iterations
	grad = ones(size(inputs, 1), 1);   % initial change in cost function (initial value is infinity)
	grad_tol = 1e-3;    % algorithm stops if change in cost is less than this in consecutive steps
	learningRate = 0.1;

	iter = 0;
	while((iter < max_iter) && (norm(grad) > grad_tol))
		grad = inputs'*(outputs - sigmoid(inputs*theta));
		theta = theta + learningRate*grad;
		iter = iter+1;
	end
end

%% sigmoid: returns sigmoid of a value / vector
function [y] = sigmoid(x)
	y = 1./ (1 + exp(-x));
	return
end 

