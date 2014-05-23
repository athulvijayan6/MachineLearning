% @Author: Athul Vijayan
% @Date:   2014-05-23 22:05:13
% @Last Modified by:   Athul Vijayan
% @Last Modified time: 2014-05-23 23:06:40

load  'iris.mat'

in = inputs(1:100,:);
out = clusters(1:100);

p = logreg(in, out);
