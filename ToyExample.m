% A toy example of quantized pyramid Weisfeiler-Lehman graph representation.
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

clear;

% Created by toyExampleHighDimension(numGraphsPerClass, nnodes, knn, mu, sigma);
% with the default values from package ToyExample
load('MultiDimensionalToyExample');

%% In order to plot graphs uncomment the follow commands
%addpath(genpath('./ToyExample'));
%for i = 1:length(Graphs)
%    figure;
%    plotKnnGraph(Graphs(i), 1);
%end

%% Set parameter k - the depth of subtree patterns of the Weisfer-Lehman
%% algorithm
kWLMax = 3;

%% Create subtree pattern features and the sequence of Weisfeiler-Lehman kernels
[WLkernels, WLpyramidFix, WLPhi, WLPhiLabels, WLPhiSz] = WLpyramid(Graphs, graphsNnodes, kWLMax, 5);

%% Run Multiple Kernel Learning with cross validation 
%% over the sequence of Weisfeiler-Lehman kernels
% Add path to Simple MKL package -- SET CORRECTLY
addpath(genpath('./SimpleMKL'));
% Add path to SVM and Kernel method package -- SET CORRECTLY
addpath(genpath('./svm-km'));

accuracyMKL = cell(1, kWLMax+1);
errbarMKL = cell(1, kWLMax+1);
accuracyFix = cell(1, kWLMax+1);
errbarFix = cell(1, kWLMax+1);
accuracyLevel = cell(1, kWLMax+1);
errbarLevel = cell(1, kWLMax+1); 
c = 0.01;
numFolds = 10;

for kWL = 0:kWLMax    
    disp(['WL kernel iteration level = ', num2str(kWL)]);
    % Run Multiple Kernel Learning over the sequence of WL kernels
    [accuracyMKL{kWL+1}, errbarMKL{kWL+1}] = WLpyramidSimpleMKL(WLkernels(:, :, kWL+1, :),classLabels, c, numFolds);
    % Run SVM on the fixed weight scheme WL pyramid kernel
    [accuracyFix{kWL+1}, errbarFix{kWL+1}] = SVMMKcrossvalidate(WLpyramidFix(:, :, kWL+1), classLabels, c, numFolds);  
   
    % Run SVM on WL kernel on per quantization level
    level = size(WLkernels, 4);
    accuracyLevel{kWL+1} = zeros(1, level);
    errbarLevel{kWL+1} = zeros(1, level);
    for l = 1:level
        [accuracyLevel{kWL+1}(l), errbarLevel{kWL+1}(l)] = SVMMKcrossvalidate(WLkernels(:, :, kWL+1, l), classLabels, c, numFolds);    
    end
end

%% Plotting cross validation accuracy
for kWL = 0:kWLMax
     figure
     errorbar([accuracyMKL{kWL+1} accuracyFix{kWL+1} accuracyLevel{kWL+1}], ...
              [errbarMKL{kWL+1} errbarFix{kWL+1} errbarLevel{kWL+1}], '*');
     hold on;
     errorbar([accuracyMKL{kWL+1} accuracyFix{kWL+1}] , ...
              [errbarMKL{kWL+1} errbarFix{kWL+1}] , '*g');
     errorbar([accuracyMKL{kWL+1}] , ...
              [errbarMKL{kWL+1}] , '*r');    
     st = sprintf('Experiment Subset - kWL is %d', kWL);
     title(st);
     ylabel('Accuracy');
     sx = sprintf('WLpyramidMKL - red, WLpyramidFix - green, WLpyramid - blue (coarser to finner)');
     xlabel(sx);
     hold off;
end


%% End of File