% @Author: athul
% @Date:   2014-05-13 10:05:12
% @Last Modified by:   athul
% @Last Modified time: 2014-05-22 20:09:19

%% LMS: function description
function [theta] = stochasticLMS(trainingInputs, trainingOutputs, learningRate, bias)

	% In this algorithm, we repeatedly run through the training set, and each time
	% we encounter a training example, we update the parameters according to
	% the gradient of the error with respect to that single training example only.
	
	if nargin == 2  % take care of optional arguments
		learningRate = 0.1;
		bias = 'bias';
	elseif nargin == 3
		bias = 'bias';
	end			

	max_iter = 100;     % maximum iterations
	delta_cost = inf;   % initial change in cost function (initial value is infinity)
	cost_tol = 0.01;    % algorithm stops if change in cost is less than this in consecutive steps

	if bias=='bias'
		theta = rand(size(trainingInputs,2) + 1,1);  % initial theta with 'bias' term
		trainingInputs = [ones(size(trainingInputs,1),1)  trainingInputs];  % initial theta with 'bias' term
	else
		theta = rand(size(trainingInputs,2),1);   % initial theta without bias term
	end

	iter = 0;
	while((iter < max_iter) && (delta_cost > cost_tol))
		cost = (trainingInputs*theta - trainingOutputs)'*(trainingInputs*theta - trainingOutputs); %cost before iter
		thetaLast = theta;
		for sample=1:size(trainingInputs,1)
			for j =1: size(theta)
				% find gradient by stochastic descent
				grad = (trainingOutputs(sample,:) - trainingInputs(sample,:) * theta)*trainingInputs(sample, j);
				theta(j) = theta(j) + learningRate*grad; % update theta for minimizing cost function
			end
		end
		newCost = (trainingInputs*theta - trainingOutputs)'*(trainingInputs*theta - trainingOutputs);
		delta_cost = abs(cost - newCost); % change in cost function due to iteration
		iter = iter+1;
	end
end

