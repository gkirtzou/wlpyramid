% Plot a knn graph in 2D 
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
function plotKnnGraph(G, v)
% Plot a knn graph in 2D 
% Input: G - the graph
%        G.am - is the adjacency matric of the graph
%        G.al - is the adjacency list of the graph
%        G.nl.coord - an 2xn array containing the 2D coordinates per node
%        G.nl.data - a dxn array containing the d dimensional data per
%                    node. The d dimensional data were selected randomly
%                    from one of the length(m) multivariate gaussian
%                    distributions
%        G.nl.values -  a column vector of node labels for the graph
%        v - a boolean: 1 plots node discrete label, 0 plots node data

    if(nargin<2)
        v = 1;
    end

    % Plot Nodes
    if size(G.nl.coord, 2) == 2
        plot(G.nl.coord(:, 1), G.nl.coord(:,2), '*')
    elseif size(G.nl.coord, 2) == 3
        plot3(G.nl.coord(:, 1), G.nl.coord(:,2), G.nl.coord(:, 3), '*')
    end
    hold on;
    if v == 0
        % Plot Data
        text( G.nl.coord(:, 1)+0.01, G.nl.coord(:,2)+0.01, num2str(floor(G.nl.data(:, :))))
    else
        % Plot Data
        text( G.nl.coord(:, 1)+0.01, G.nl.coord(:,2)+0.01, num2str(G.nl.values(:)))
    end
    % Plot Edges
    nnodes = length(G.am);
    for i = 1:nnodes 
        e = find(G.am(i, :)==1);
        for j = 1:length(e)
            if e(j) < i
                continue;
            end
            if size(G.nl.coord, 2) == 2
                plot([G.nl.coord(i,1), G.nl.coord(e(j),1)], [G.nl.coord(i,2), G.nl.coord(e(j),2)], '-')
            elseif size(G.nl.coord, 2) == 3
                plot3([G.nl.coord(i,1), G.nl.coord(e(j),1)], [G.nl.coord(i,2), G.nl.coord(e(j),2)], ...
                      [G.nl.coord(i,3), G.nl.coord(e(j),3)], '-')
            end
        end
    end
    hold off;
end
%% End Of File
