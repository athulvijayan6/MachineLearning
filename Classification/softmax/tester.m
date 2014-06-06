% @Author: Athul Vijayan
% @Date:   2014-06-06 17:56:22
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-06-06 22:58:54


load 'iris.mat';
theta = softmax([inputs(1:33,:) ; inputs(51:83,:); inputs(116:137,:)], [clusters(1:33) ; clusters(51:83); clusters(116:137)])
clusterCount = max(clusters);

for i=84:115
	for l = 1:clusterCount
		P(l) = exp(theta(l, :)*[1 inputs(i,:)]');
	end
	P = P ./ sum(P)
	[k,p] = max(P);
	disp 'predicted'
	p
	disp 'actual'
	clusters(i)
end