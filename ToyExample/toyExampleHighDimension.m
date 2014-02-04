% Create a simple toy example of knn graphs, that consists of two class in
% the unit square. 
% Author: Katerina Gkirtzou, Matthew Blaschko
% Copyright 2012-2013 Katerina Gkirtzou, Matthew Blaschko
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

function [Graphs, ClassLabel] = toyExampleHighDimension(numGraphsPerClass, nnodes, knn, mu, sigma)
% Create a simple toy example of knn graphs 
% The toy example consists of two class in the unit square.  
% Input: numGraphsPerClass - a 2D vector, with the number of wanted graphs
%                            per distribution, default value = [100 100]
%        nnodes - scalar, the number of nodes per graph, 
%                   default value = 100
%        knn - scalar, the number of neighbors per node (knn graph),
%              default value = 5
%        mu - matrix of size 2xd, containing the mean for creating 2 
%             d-dimensional gaussian distributions, default value = [0 0; 10 10]
%        sigma - cell list of size 2, containing the covariance for creating 2
%                d-dimensional gaussian distributions, default value = [5,
%                0; 0 5] for each distribution
% Output: Graphs - a vector of graphs of size sum(numGraphPerClass)
%         ClassLabel - a vector of scalars of size sum(numGraphsPerClass),
%                      containing the class label of each graph
% Requires the KD-Tree library of Andrea Tagliasacchi
% Link: % https://sites.google.com/site/andreatagliasacchi/software/matlabkd-treelibrary
    
    % Set default values, in case arguments are missing
    if nargin < 5
        sigma = cell(1, 2);
        sigma{1} = [5 0; 0 5];
        sigma{2} = [5 0; 0 5];
    end

    if nargin < 4 
        mu = [0 0; 10 10];
    end

    if nargin < 3
        knn = 5;
    end

    if nargin < 2
        nnodes = 100;
    end

    if nargin < 1 
        numGraphsPerClass = [100, 100];
    end

    % Check correct sizes
    if size(mu, 1) ~= size(sigma, 2)
        error('Number of mean''s array and Covariances''s array does not have the same size\n');
    end
    
    if size(mu, 2) ~= size(sigma{1}, 2)
         error('Number of means for a multivariate distribution and covariances''s size is not have the same\n');
    end

    if size(numGraphsPerClass, 2) ~= 2
         error('NumGraphsPerClass is not a 2D vector\n');
    end

    dimension = size(mu, 2);
    % Create graphs for positive class
    for i = 1:numGraphsPerClass(1)
        % Create knn graph structure
        posGraphs(i) = createKnnGraph(nnodes, knn);

        % Add data to nodes
        % Select nodes that will have data from Distribution 1
        nodesDist1 = posGraphs(i).nl.coord(:,1) < 0.5; 
        posGraphs(i).nl.data = zeros(nnodes, dimension);
        % Set data to nodes from Distribution 1
        posGraphs(i).nl.data(find(nodesDist1 == 1), :) = mvnrnd(mu(1, :), sigma{1}, length(find(nodesDist1 == 1)));
        % Set data to nodes from Distribution 2
        posGraphs(i).nl.data(find(nodesDist1 == 0), :) = mvnrnd(mu(2, :), sigma{2}, length(find(nodesDist1 == 0)));

        % Preparation step for pyramid labelling
        % Sort data per dimension 
        posGraphs(i).nl.sortedData = zeros(nnodes, dimension);
        posGraphs(i).nl.sortedIndex = zeros(nnodes, 1);
        [posGraphs(i).nl.sortedData, posGraphs(i).nl.sortedIndex(:)] = sortrows(posGraphs(i).nl.data);

        % Set labels corser level
        posGraphs(i).nl.values = ones(nnodes, 1);
        % Add labels 1 and 2 to nodes just for plotting 
        posGraphs(i).nl.values(find(nodesDist1 == 1)) = 1;
        posGraphs(i).nl.values(find(nodesDist1 == 0)) = 2;
    end


    % Create graphs for negative class
    for i = 1:numGraphsPerClass(2)
        % Create knn graph structure
        negGraphs(i) = createKnnGraph(nnodes, knn);

        % Add data to nodes
        % Select nodes that will have data from Distribution 1
        tmp = ((1-sqrt(0.5))/2);

        nodesDist1 = negGraphs(i).nl.coord(:,1) > tmp & ...
                     negGraphs(i).nl.coord(:,2) > tmp & ...
                     negGraphs(i).nl.coord(:,1) < (1-tmp) & ...
                     negGraphs(i).nl.coord(:,2) < (1-tmp);

        negGraphs(i).nl.data = zeros(nnodes, dimension);
        % Set data to nodes from Distribution 1
        negGraphs(i).nl.data(find(nodesDist1 == 1), :) = mvnrnd(mu(1, :), sigma{1}, length(find(nodesDist1 == 1)));
        % Set data to nodes from Distribution 2
        negGraphs(i).nl.data(find(nodesDist1 == 0), :) = mvnrnd(mu(2, :), sigma{2}, length(find(nodesDist1 == 0)));

        % Preparation step for pyramid labelling
        % Sort data per dimension 
        negGraphs(i).nl.sortedData = zeros(nnodes, 1);
        negGraphs(i).nl.sortedIndex = zeros(nnodes, 1);
        [negGraphs(i).nl.sortedData, negGraphs(i).nl.sortedIndex(:)] = sortrows(negGraphs(i).nl.data);

        % Set labels corser level
        negGraphs(i).nl.values = ones(nnodes, 1);
        % Add labels 1 and 2 to nodes just for plotting 
        negGraphs(i).nl.values(find(nodesDist1 == 1)) = 1;
        negGraphs(i).nl.values(find(nodesDist1 == 0)) = 2;
    end

    Graphs = [posGraphs, negGraphs];
    ClassLabel = [ones(numGraphsPerClass(1),1); -ones(numGraphsPerClass(2),1)];
end

%% End Of File