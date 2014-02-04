% Trains a SVM given a custom kernel K, using numFolds cross validation
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

function [accuracy, errorbar] = SVMMKcrossvalidate(K, classLabels, c, numFolds, cvInd)
% Trains a SVM given a custom kernel K, using numFolds cross validation
% Author: Katerina Gkirtzou
% Copyright: Ecole Centrale Paris 2012
% Input: K - a nxn kernel matrix \in \mathbb{R}^{n \times n}
%        classLabels - a label vector \in \{-1, +1\}^{n}
%        c - the SVM C parameter \in \mathbb{R}_{+}
%        numFolds - don't need to pass this in, will default to 10
% Requires the SVM and Kernel Methods Matlab Toolbox
% Link: http://asi.insa-rouen.fr/enseignants/~arakoto/toolbox/index.html

     %% Setting Default values
    if(nargin<3)
        c = 1e4/size(K, 1);
    end

    if(nargin<4)
        numFolds = 10;
    end
    
    %% Setting svm kernel's parameters
    kernel = 'numerical';         % type of kernel
    lambdareg = 1e-8;             % ridge added to kernel matrix for QP method
    verbosesvm = 0;                % verbosity of inner svm algorithm 
   % span = 1;                     % span matrix for semiparametric learning
   % alphainit = [];

    
    %% Split data for cross validation
    if nargin < 5
        cvInd = crossvalind('kfold', classLabels, numFolds);
    end
    accuracyCV = zeros(1, numFolds);
    for i=1:numFolds
        %% Train
        indTrain = find(cvInd ~= i);
        kerneloption.matrix = K(indTrain, indTrain);  % set custom kernel 
        classLabelsTrain = classLabels(indTrain);
        [supVec,weightSupVec,bias,indSupVec,aux,aux,obj] = svmclass([],classLabelsTrain,c,lambdareg,kernel,kerneloption,verbosesvm);%,span,alphainit)
        
        %% Test
        indTest = find(cvInd == i);
        KTest = K(indTest, indTrain(indSupVec));
        classLabelsPred=KTest*weightSupVec + bias;
        classLabelsPred(find(classLabelsPred == 0)) = -1;
        classLabelsTest = classLabels(indTest);

        accuracyCV(i)=sum(sign(classLabelsPred)==classLabelsTest)/length(classLabelsTest);       
    end
    accuracy = mean(accuracyCV);
    errorbar = std(accuracyCV);

end