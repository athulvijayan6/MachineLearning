% @Author: athul
% @Date:   2014-05-13 10:05:12
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-05-23 23:12:54

%% LMS: function description
function [theta] = batchLMS(trainingInputs, trainingOutputs, learningRate, bias)
	
	% This method looks at every example in the entire training set on every step, and is called batch
	% gradient descent.

	if nargin == 2      % take care of optional arguments
		learningRate = 0.04;
		bias = 'bias';
	elseif nargin == 3
		bias = 'bias';
	end			

	max_iter = 100000;     % maximum iterations
	grad = ones(size(trainingInputs, 1), 1);   % initial change in cost function (initial value is infinity)
	grad_tol = 1e-3;    % algorithm stops if change in cost is less than this in consecutive steps

	if bias=='bias'
		trainingInputs = [ones(size(trainingInputs,1),1)  trainingInputs];    % add 1 since we have 'bias'  term
	end
	theta = rand(size(trainingInputs,2),1);  % initial theta without bias term
	
	iter = 0;
	while((iter < max_iter) && (norm(grad) > grad_tol))
		grad = trainingInputs'*(trainingInputs*theta - trainingOutputs);
		theta = theta - learningRate*grad;
		iter = iter+1;	
	end
end


