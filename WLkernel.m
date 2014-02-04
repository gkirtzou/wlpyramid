% Compute k-step Weisfeiler-Lehman kernel for a set of graphs.
% Author: Nino Shervashidze, Katerina Gkirtzou
% Copyright 2012-2013 Nino Shervashidze, Katerina Gkirtzou
%
% This file is part of the WLpyramid package
% 
% WLpyramid is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% WLpyramid is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with WLpyramid.  If not, see
% <http://www.gnu.org/licenses/>.

function [K, PhiAll, labelsAll, runtime] = WLkernel(Graphs,k,nl, kerneltype)
% Compute k-step Weisfeiler-Lehman kernel for a set of graphs
% Author: Nino Shervashidze, Katerina Gkirtzou 
% Input: Graphs - a 1xn_graphs array of graphs
% 		  Graphs(i).am is the adjacency matrix of the i'th graph,
% 		  Graphs(i).al is the adjacency list of the i'th graph,
%         Graphs(i).nl.values is a column vector of node
%                 labels for the i'th graph.
%         Graphs(i) may have other fields, but they will not be
%                 used here.
%	 k - a natural number: number of iterations of WL
%	 nl - a boolean: 1 if we want to use original node labels, 0 otherwise
%    kernertype - a string that describes the kernel type of the WL
%    algorithm. Two possible options 'intersection' (default) or 'linear'
% Output: K - a nxnx(k+1) array containing all k+1 step Weisfer-Lehman nxn
%             kernel matrices
%         runtime - scalar (runtime in seconds)

if nargin < 4
    kerneltype = 'histogram';
end

n_graphs=size(Graphs,2);
Lists = cell(1,n_graphs);
N=0;
% compute adjacency lists and N, the total number of nodes in the dataset
for i=1:n_graphs
    Lists{i}=Graphs(i).al;
    N=N+size(Graphs(i).am,1);
end
%N
phi=sparse(N,n_graphs); %each column j of phi will be the explicit feature
% representation for the graph j

t=cputime; % for measuring runtime
K = zeros(n_graphs, n_graphs, k+1);
PhiAll = cell(1, k+1);
labelsAll = cell(1, k+1);
disp(['WL interation 0 - ', datestr(now)]);
%%% INITIALISATION
% initialize the node labels for each graph with their labels or
% with degrees (for unlabeled graphs). This is also the first iteration of WL
if nl==1
    % label_lookup is an associative array, which will contain the
    % mapping from multiset labels (strings) to short labels (integers)
    label_lookup=containers.Map();
    label_counter=uint32(1);
    for i=1:n_graphs
        % the type of labels{i} is uint32, meaning that it can only handle
        % 2^32 labels and compressed labels over all iterations. If
        % more is needed, switching (all occurences of uint32) to
        % uint64 is a possibility
        labels{i}=zeros(size(Graphs(i).nl.values,1),1,'uint32');
        for j=1:length(Graphs(i).nl.values)
            str_label=num2str(Graphs(i).nl.values(j));
            % str_label is the node label of the current node of the
            % current graph converted into a string
            if ~isKey(label_lookup, str_label)
                label_lookup(str_label)=label_counter;
                label_counter=label_counter+1;
            end
            labels{i}(j)=label_lookup(str_label);
            phi(labels{i}(j),i)=phi(labels{i}(j),i)+1;
        end
    end
else
    for i=1:n_graphs
        labels{i}=uint32(full(sum(Graphs(i).am,2)));
        for j=1:length(labels{i})
            phi(labels{i}(j)+1,i)=phi(labels{i}(j)+1,i)+1;
        end
    end
end
clear Graphs;
h=1;
if strcmp(kerneltype , 'linear')
    K(:, :, h) =full(phi'*phi);
elseif strcmp(kerneltype, 'histogram')
    K(:, :, h) = histogramIntersect(phi);
end

if nargout >= 2
    PhiAll{h} = phi';
end
if nargout >= 3 
    labelsAll{h} = labels;
end

%%% MAIN LOOP
new_labels=labels;
while h<=k
    disp(['WL interation h = ',num2str(h), ' - ', datestr(now)]);
    % create an empty lookup table
    label_lookup=containers.Map();
    label_counter=uint32(1);
    % create a sparse matrix for feature representations of graphs
    phi=sparse(N,n_graphs);
    for i=1:n_graphs
        for v=1:length(Lists{i})
            % form a multiset label of the node v of the i'th graph
            % and convert it to a string
            long_label=[labels{i}(v), sort(labels{i}(Lists{i}{v}))'];
            long_label_2bytes=typecast(long_label,'uint16');
            long_label_string=char(long_label_2bytes);
            % if the multiset label has not yet occurred, add it to the
            % lookup table and assign a number to it
            if ~isKey(label_lookup, long_label_string)
                label_lookup(long_label_string)=label_counter;
                new_labels{i}(v)=label_counter;
                label_counter=label_counter+1;
            else
                new_labels{i}(v)=label_lookup(long_label_string);
            end
        end
        % fill the column for i'th graph in phi
        for j=1:length(labels{i})
            phi(new_labels{i}(j),i)=phi(new_labels{i}(j),i)+1;
        end
    end
    %K=K+full(phi'*phi);
    if strcmp(kerneltype, 'linear')
        K(:, :, h+1) = K(:, :, h) + full(phi'*phi);
    elseif strcmp(kerneltype, 'histogram')
        K(:, :, h+1) = K(:, :, h) + histogramIntersect(phi);
    end
    
    if nargout >= 2
        PhiAll{h+1} = phi';
    end
    if nargout >=3
        labelsAll{h+1} = new_labels;
    end
    labels=new_labels;
    h=h+1;
end
runtime=cputime-t; % computation time of K

end

function K = histogramIntersect(phi)
% Computes the histogram Intersection kernel.
% Author: Katerina Gkirtzou 
% Input: phi - n vector of histograms of features 
% Output: K - nxn intersection kernel
% 
    for i=1:size(phi,2)
        for j=i:size(phi,2)
            K(i,j) = sum(min(phi(:,i),phi(:,j)));
            K(j,i) = K(i,j);
        end
    end
end

%% End Of File

