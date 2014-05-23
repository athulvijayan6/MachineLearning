% @Author: athul
% @Date:   2014-05-22 21:58:19
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-05-23 10:11:13

%% loess: function description
function [theta] = loess(trainingInputs, trainingOutputs, x, tau)
	if nargin == 3
		tau = 1;
	end

	theta = rand(size(trainingInputs,2) + 1,1);  % initial theta with 'bias' term
	trainingInputs = [ones(size(trainingInputs,1),1)  trainingInputs];  % initial theta with 'bias' term

	max_iter = 100;     % maximum iterations
	delta_cost = inf;   % initial change in cost function (initial value is infinity)
	cost_tol = 0.01;    % algorithm stops if change in cost is less than this in consecutive steps
	learningRate = 0.1;

	iter = 0;
	while((iter < max_iter) && (delta_cost > cost_tol))
		cost = 0;
		for i=1: size(trainingOutputs)
			cost = cost + weight(trainingInputs(i,:), x, tau)*(trainingInputs(i, :)*theta - trainingOutputs(i))^2;
		end

		for sample=1:size(trainingInputs,1)
			for j =1: size(theta)
				% find gradient by stochastic descent
				grad = weight(trainingInputs(sample,:), x, tau)*(trainingOutputs(sample,:) - trainingInputs(sample,:) * theta)*trainingInputs(sample, j);
				theta(j) = theta(j) + learningRate*grad; % update theta for minimizing cost function
			end
		end
		newCost = 0;
		for i=1: size(trainingOutputs)
			newCost = newCost + weight(trainingInputs(i,:), x, tau)*(trainingInputs(i, :)*theta - trainingOutputs(i))^2;
		end
		delta_cost = abs(cost - newCost); % change in cost function due to iteration
		iter = iter+1;
	end	

end

%% weight: Finds weight at x
function [w] = weight(a, b, tau)
	if nargin == 2
		tau = 1;
	end
	w = exp(-(a-b)*(a-b)'/(2*tau^2));
	return
end
