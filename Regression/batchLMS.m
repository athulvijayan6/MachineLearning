% @Author: athul
% @Date:   2014-05-13 10:05:12
% @Last Modified by:   athul
% @Last Modified time: 2014-05-22 20:10:29

%% LMS: function description
function [theta] = batchLMS(trainingInputs, trainingOutputs, learningRate, bias)
	
	% This method looks at every example in the entire training set on every step, and is called batch
	% gradient descent.

	if nargin == 2      % take care of optional arguments
		learningRate = 0.001;
		bias = 'bias';
	elseif nargin == 3
		bias = 'bias';
	end			

	max_iter = 100;     % maximum iterations
	delta_cost = inf;   % initial change in cost function (initial value is infinity)
	cost_tol = 0.01;    % algorithm stops if change in cost is less than this in consecutive steps

	if bias=='bias'
		theta = rand(size(trainingInputs,2) + 1,1);  % initial theta with 'bias' term
		trainingInputs = [ones(size(trainingInputs,1),1)  trainingInputs];    % add 1 since we have 'bias'  term
	else
		theta = rand(size(trainingInputs,2),1);  % initial theta without bias term
	end
	
	iter = 0;
	while((iter < max_iter) && (delta_cost > cost_tol))
		cost = (trainingInputs*theta - trainingOutputs)'*(trainingInputs*theta - trainingOutputs);%cost before iter
		for j =1: size(theta)
			grad = (trainingInputs*thetaLast - trainingOutputs)'*trainingInputs(:, j); % find gradient by batch descent
			theta(j) = theta(j) - learningRate*grad;    % update theta for minimizing cost function
		end
		newCost = (trainingInputs*theta - trainingOutputs)'*(trainingInputs*theta - trainingOutputs);
		delta_cost = abs(cost - newCost); % change in cost function due to iteration
		iter = iter+1;
	end
end


