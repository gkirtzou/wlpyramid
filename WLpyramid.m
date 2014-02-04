% The pyramid quantized Weisfeiler-Lehman graph representation 
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

function [WLpyramidKernels, WLpyramidFix, Phi, PhiLabels, PhiSz] = WLpyramid(Graphs, nnodes, kWL, levels, kernelType, normalizeFlag)
% The pyramid quantized Weisfeiler-Lehman graph representation
% Input: Graphs - 1xn array of n graphs
%        nnodes - 1xn array containing the number of nodes per graph
%        kWL - a scalar, the depth of subtree patterns
%        levels - a scalar, the number of pyramid levels of quantization.
%                 Default value: the log2 of the unique values of the node
%                 labels. If levels = 0, it also uses the default value.
%        kernertype - a string that describes the kernel type of the WL
%                     algorithm. Two possible options 'histogram' or 'linear'
%                     Default value: 'histogram'
%        normalizeFlag - a flag, one normalizes the columns (variables) of
%                 a label matrix to unit Euclidean length, zero otherwise
%                 Default value: zero
% Output: WLpyramidKernels - a nxnx(k+1)x(levels+1) array of the Weisfeiler-Lehman kernel
%                            for all k+1 depths of subtree patterns and for all
%                            pyramid quantization levels
%         WLpyramidFix - a nxnx(k+1) array with the fix weighted scheme of the
%                        Weisfeler-Lehman kernel for all k+1 depths of
%                        subtree patterns.
%         Phi - a  nx(PhiSz*(levels+1)) array of histograms of subtree
%               patterns appearances for all pyramid quantization levels. 
%         PhiLabels - a PhiSz*(levels+1) vector with the subtree pattern
%                     (short subtree pattern label as created by the Weisfeiler-Lehman
%                     algorithm).
%         PhiSz - a scalar with the size of the subtree pattern alphabet.
%

    if nargin < 6
       normalizeFlag = 0;
    end
    
    if nargin < 5
        kernelType = 'histogram';
    end
    
    n = length(Graphs); % number of Graphs
    if length(nnodes) ~= length(Graphs)
        error('Nnodes should have the same size as Graphs');
    end

    %% Gather all data from all nodes in one single array
    % sum(nnodes) : the number of nodes of all graphs
    % size(Graphs(1).nl.data, 2) : the dimension of the data
    data = zeros(sum(nnodes), size(Graphs(1).nl.data, 2));
    indexLimit = zeros(1, n);
    for i = 1:n
        indexLimit(i) = sum(nnodes(1:i));
        data(1+sum(nnodes(1:i-1)):indexLimit(i), :) = Graphs(i).nl.data;
    end
    
    if normalizeFlag
       [data mu d] = normalize(data);         
    end
    
    %% Quantization of the continuous vector label space
    maxSize = 1e4; 
    disp(['Quantization of continuous vector label space - ', datestr(now)]);
    data = real(data);
    disp(['Labels size [', num2str(size(data, 1)), ',' num2str(size(data, 2)) , ']']);
    % If labels space is small use it all
    if size(data, 1) <= maxSize
        % Number of pyramid quantization levels
        if nargin < 4
            levels = ceil(log2(size(unique(data, 'rows'), 1)));
        end
        if exist('levels', 'var') && levels == 0
            levels = ceil(log2(size(unique(data, 'rows'), 1)));
        end
        % Labels per node across all pyramid levels
        % size(dataLabels) == sum(nnodes)xlevels;
        Z = linkage(data, 'ward', 'euclidean');
        % lower level number bigger number of bins 
        % due to cluster function 
        labels = 2.^[0:levels];
        dataLabels = cluster(Z, 'maxclust', labels);
    else
        % Select randomly some data to represent the feature space
        indexSel = randsample(size(data, 1), maxSize);
        dataSel = data(indexSel, :);
        if nargin < 4
            levels = ceil(log2(size(unique(dataSel, 'rows'), 1)));
        end
        if exist('levels', 'var') && levels == 0
            levels = ceil(log2(size(unique(dataSel, 'rows'), 1)));
        end
        % agglomerative way
        disp(['Creating pyramid quantization  - ',  datestr(now)]);
        labels = 2.^[0:levels];
        Z = linkage(dataSel, 'ward', 'euclidean');
        T = cluster(Z, 'maxclust', labels); 
        clustMeans = cell(1, length(labels));
        for l = 1:(levels+1)
            disp(['Finding labels of quantization at level ', num2str(l), ' - ', datestr(now)]);
            M = zeros(size(dataSel,1),max(T(:, l)));
            M(sub2ind(size(M),[1:size(M,1)]',T(:, l))) = 1;
            clustMeans{l} = diag(1./sum(M))*M'*dataSel;
        end
        % Get discrete labels in chunks
        disp(['Getting discrete labels - ', datestr(now)]);
        dataLabels = zeros(size(data, 1), length(labels));
        nChunks = floor(size(data, 1)/maxSize)-1;
        for i = 1:nChunks
            for l =  1:(levels+1)
                distances = pdist2(data((i-1)*maxSize+1:i*maxSize, :), clustMeans{l});
                [minD, clusterAssignments] = min(distances, [], 2);
                dataLabels((i-1)*maxSize+1:i*maxSize, l) = clusterAssignments;
            end
        end
        if ~isempty(data(i*maxSize+1:end, :))
            for l =  1:(levels+1)
                distances = pdist2(data(i*maxSize+1:end, :), clustMeans{l});
                [minD, clusterAssignments] = min(distances, [], 2);
                dataLabels(i*maxSize+1:end, l) = clusterAssignments;
            end
        end
    end
       
    %% Create Weisfeler-Lehman kernels
    disp(['Creating Weisfeler-Lehman kernels for all ', num2str(levels+1), ' pyramid levels - ', datestr(now)]);
    previousKernel = zeros(n, n, kWL+1);
    WLpyramidFix = zeros(n, n, kWL+1);
    WLpyramidKernels = zeros(n, n, kWL+1, levels+1);
    Phi = cell(1, kWL+1);
    PhiLabels = cell(levels+1, kWL+1);
    % Note:: The higher the level number the smaller number of quantization bins
    for l = 1:(levels+1)
        % Get discrete labels for level l
        disp(['WLpyramid kernel for level l = ', num2str(l)]);
        for i = 1:n
            Graphs(i).nl.values = dataLabels(1+sum(nnodes(1:i-1)):indexLimit(i), l);
        end
        if nargout > 2
           [currentKernel, currPhi, currLabels] = WLkernel(Graphs, kWL, 1, kernelType); 
        else
            currentKernel = WLkernel(Graphs, kWL, 1, kernelType); 
        end
        WLpyramidFix(:, :, :) = WLpyramidFix(:, :, :) + 1/2^(l).*(currentKernel - previousKernel);
        WLpyramidKernels(:, :, :, l) = currentKernel;
        if nargout >= 4
            for i = 1:kWL+1
                Phi{i} = [Phi{i}, currPhi{i}];
                PhiLabels{l, i} = currLabels{i};
            end
            if l == levels
                PhiSz = size(currPhi{1}, 2);
            end
        end
        previousKernel = currentKernel;
    end
end


function [X mu d] = normalize(X)
% Centers and scales the observations of a data matrix, such that each variable (column) has
% unit Euclidean length.
% Input: X - mxn array of m observations of n variables. 
% Output: X - mxn array of normalized m observations of n variables. 
%        mu - n vector with the mean value for each variable
%        d - n vector with the Euclidean lengths for each variable 
%

    n = size(X,1);
    if nargin < 2
        mu = mean(X);
        X = X - ones(n,1)*mu;
    else
        X = X - ones(n,1)*mu;
    end

    if nargin < 3
        d = sqrt(sum(X.^2));
        d(d == 0) = 1;
    end

    X = X./(ones(n,1)*d)

end

%% End Of File





