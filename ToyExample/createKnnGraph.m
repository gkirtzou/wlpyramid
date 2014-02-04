% Create a knn graph in 2D as a toy example for the quantized pyramid
% Weisfeiler-Lehman graph representation. 
% Author: Katerina Gkirtzou
% Copyright 2012-2013 Katerina Gkirtzou
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

function G = createKnnGraph(nnodes, k)
% Create a knn graph in 2D as a toy example for the quantized pyramid
% Weisfeiler-Lehman graph representation. 
% Input: nnodes - the number of nodes of the graph
%        k - scalar, the number of neighbors per nodes (knn graph)
% Output: G - the graph
%         G.am - is the adjacency matric of the graph
%         G.al - is the adjacency list of the graph
%         G.nl.coord - an nx2 array containing the 2D coordinates per node
% Requires the KD-Tree library of Andrea Tagliasacchi
% Link: https://sites.google.com/site/andreatagliasacchi/software/matlabkd-treelibrary

    % Create nnodes nodes in 2D in space [0,1]x[0,1]
    % for visualization reasons
    G.nl.coord = rand(nnodes, 2);
    % Create adjecency matrix of the graph
    G.am = zeros(nnodes, nnodes);
    % Create the KD tree
    kdtree = kdtree_build(G.nl.coord);
    % Find the k nearest neighbord per node
    for i = 1:nnodes
        idxs = kdtree_k_nearest_neighbors(kdtree, G.nl.coord(i,:), k); 
        G.am(i, idxs) = 1;
        G.am(idxs, i) = 1;
        G.am(i, i) = 0;
    end
    kdtree_delete(kdtree);
    G.al = adjacencyList(G.am);

end

function [al] = adjacencyList(am)
% Input: am - nxn adjacency matrix
% Output: al - 1xn cell array of vectors: corresponding adjacency list

    n = size(am,1); % number of nodes

    al=cell(n,1);
    for i = 1:n
        al{i} = find(am(i,:)==1);
    end

end

%% End Of File







