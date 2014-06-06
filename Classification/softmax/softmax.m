% @Author: Athul Vijayan
% @Date:   2014-05-25 10:31:30
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-06-07 00:46:23

%% softmax: function does a softmax regression for classification
% The function computes the parametrs for the softmax regression which is a classification algorithm.
% The outputs have to belong to {1,2,3...,k} where k is highest labels.
% For making a prediction,
% 		yhat = theta*testPoint'
% which will give probabilities of each label. classify according to max probability.
% optional parameter lamda, for weight decay term. Igmore this to omit weight decay
function [theta] = softmax(inputs, outputs, lambda)
	if nargin == 2
		lambda = 0;
	end
	clusterCount = max(outputs);
	m = size(inputs, 1);
	inputs = [ones(size(inputs, 1), 1) inputs];
	theta = rand(clusterCount, size(inputs, 2));

	max_iter = 100;     % maximum iterations
	grad = ones(clusterCount, size(inputs, 2));
	grad_tol = 1e-3;    % algorithm stops if change in grad is less than this in consecutive steps
	step = 0.08;
	iter = 0;

	while((iter < max_iter) && (norm(grad) > grad_tol))
		grad = zeros(clusterCount, size(inputs, 2));
		for j = 1:clusterCount
			for i=1:m
				for l = 1:clusterCount
					P(l) = exp(theta(l, :)*inputs(i,:)');  % -max(theta(l, :)*inputs(i, :)')
				end
				P = P ./ sum(P);
				grad(j, :) = grad(j, :) + (inputs(i, :)*((outputs(i) == j) - P(j)));
			end
			grad(j, :) = -1*(grad(j, :)./m) + lambda*theta(j, :);
		end		
		theta = theta - step * grad;
		iter = iter+1;
	end
end