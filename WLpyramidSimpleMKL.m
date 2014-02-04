% Running SimpleMKL algorithm to set the correct weights between the
% differences of consecutive pyramid levels of Weisfer-Lehman kernels
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

function  [accuracy, errorbar, preds, labels, weightKernels, weightSupVec,bias, indSupVec, cvInd] = WLpyramidSimpleMKL(WLpyramidKernels, classLabels, c, numFolds, cvInd)
% Running SimpleMKL algorithm to set the correct weights between the
% differences of consecutive pyramid levels of Weisfer-Lehman kernels
% Author: Katerina Gkirtzou
% Copyright: Ecole Centrale Paris 2012
% Input: WLpyramidKernels - a nxn1xlevel+1 array with the differences between 
%                            consecutive levels' Weisfer-Lehman kernel
%        classLabels - a label vector \in \{-1, +1\}^{n}
%        c - the SVM C parameter \in \mathbb{R}_{+}
%        numFolds - the number of folds for cross validaion, default value equals to 10
% Output: accuracy - the mean accuracy of numFolds cross validation
%         errorbar - the stantard deviation of the accuracy of numFolds
%                    cross validation
%         weighKernels - a 1xnumFolds cell list with the learned weight of
%                        SimpleMKL of the input kernels per fold
%         weightSupVec - a 1xnumFolds cell list with the weights of the 
%                        support vectors of the SVM per fold
%         bias - a 1xnumFolds array with the bias of the SVM per fold
%         indSupVec - a 1xnumFolds cell list with the index of the 
%                        support vectors of the SVM per fold
% Requires the SimpleMKL toolbox
% Link: http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html

    %% Setting Default values
    if nargin < 3
        c = 1e4/size(WLpyramidKernels(:, :, 1), 1);
    end

    if nargin < 4 
        numFolds = 10;
    end
    

    verbose=0;
    
    %------------------------------------------------------
    % Simple MKL parameters
    %------------------------------------------------------

    options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                             % 'svmclass' or 'svmreg'
    %------------------------------------------------------
    % choosing the stopping criterion
    %------------------------------------------------------
    options.stopvariation= 0; % use variation of weights for stopping criterion 
    options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
    options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

    %------------------------------------------------------
    % choosing the stopping criterion value
    %------------------------------------------------------
    options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
    options.seuildiffconstraint=0.1;    % stopping criterion for KKT
    options.seuildualitygap=0.01;       % stopping criterion for duality gap

    %------------------------------------------------------
    % Setting some numerical parameters 
    %------------------------------------------------------
    options.goldensearch_deltmax=1e-1; % initial precision of golden section search
    options.numericalprecision=1e-8;   % numerical precision weights below this value
                                       % are set to zero 
    options.lambdareg = 1e-8;          % ridge added to kernel matrix 

    %------------------------------------------------------
    % some algorithms paramaters
    %------------------------------------------------------
    options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                       % variable in the reduced gradient method 
    options.nbitermax=500;             % maximal number of iteration  
    options.seuil=0;                   % forcing to zero weights lower than this 
    options.seuilitermax=10;           % value, for iterations lower than this one 

    options.miniter=0;                 % minimal number of iterations 
    options.verbosesvm=0;              % verbosity of inner svm algorithm 
    options.efficientkernel=1;         % use efficient storage of kernels 

    %------------------------------------------------------
    
    %% Split data for cross validation
    if nargin < 5
        cvInd = crossvalind('kfold', classLabels, numFolds);
    end
    accuracyCV = zeros(1, numFolds);
    weightKernels = cell(1, numFolds);
    weightSupVec = cell(1, numFolds);
    bias = zeros(1, numFolds);
    indSupVec = cell(1, numFolds);
    preds = [];
    labels = [];
    
    for i=1:numFolds
        %% Train
        indTrain = find(cvInd ~= i); 
        KTrain = WLpyramidKernels(indTrain, indTrain, :);
        classLabelsTrain = classLabels(indTrain);
        [weightKernels{i}, weightSupVec{i}, bias(i), indSupVec{i}, story(i), obj(i)] = mklsvm(KTrain, classLabelsTrain , c, options, verbose);
               
        %% Test
        indTest = find(cvInd == i);
        KTest = zeros(length(indTest), length(indSupVec{i}));
        indK = find(weightKernels{i});
        for j = 1:length(indK)
            k = indK(j);
            KTest = KTest + WLpyramidKernels(indTest, indTrain(indSupVec{i}), k)*weightKernels{i}(k);
        end
        classLabelsPred=KTest*weightSupVec{i} + bias(i);
        classLabelsTest = classLabels(indTest);

        accuracyCV(i)=sum(sign(classLabelsPred)==classLabelsTest)/length(classLabelsTest);
        preds = [preds; classLabelsPred];
        labels = [labels; classLabelsTest];
        
    end;
    
    accuracy = mean(accuracyCV);
    errorbar = std(accuracyCV);
end

%% End Of File




