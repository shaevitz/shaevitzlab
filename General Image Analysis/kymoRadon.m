% Copyright (C) 2010, Joshua W. Shaevitz <jshaevitz@gmail.com>
% 
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details. 
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307,
% USA.

function [ang,projI,corr,theta]=kymoRadon(I,angleIncrement)

% Calcualtes single-angle back projection from a
% radon transform and finds the angle with the highest correlation to the
% input image. Inspired by Ludington and Wallace.
%
% [ang,projI,corr,theta]=kymoRadon(I,angleIncrement)
%
% INPUTS:
% I = input kymograph
% angleIncrement = number of degrees between samples
%
% OUTPUTS:
% projI = back-projected image with highest correlation
% corr = Corrlation coefficient vs angle
% theta = angle vector
% ang = the best fit angle


theta=[-90:angleIncrement:89];

corr=zeros(1,length(theta));

rI = radon(I,theta);  % Radon transform

%Optional thresholding of transform to clean up
threshold = max(max(rI))*0;
rI = max(rI-threshold,0) + threshold;

bigSize = max(size(I,1), size(I,2));

%projFig = figure;
%colormap(hot);

bar1 = waitbar(0,'Calculating correlation over angles');

for n=1:length(theta)
    waitbar(n / length(theta),bar1)
    
    rAngle = rI(:,n);           % Pickout current angle data from radon transform
    
    irI = iradon([rAngle rAngle],[theta(n) theta(n)],'spline','Ram-Lak',1,bigSize); %Calculate single angle projection
    
    if(size(I,1) > size(I,2))
        irI = irI(:, floor((size(I,1)-size(I,2))/2+2) : floor((size(I,1)-size(I,2))/2) + size(I,2)+1  );
    else
        irI = irI(floor((size(I,2)-size(I,1))/2+2) : floor((size(I,2)-size(I,1))/2) + size(I,1)+1 , : );
    end
    
    %imagesc(irI)
    %drawnow
    
    c=corrcoef(I,irI);              % calculate correlation
    
    if numel(c)==4 && isreal(c(2))
        corr(n) = c(2);
    else corr(n) = 0;
    end
    
end

%close(projFig);
close(bar1)

[maxE,i]=max(corr);

ang=theta(i);
   
rAngle = rI(:,i);

irI = iradon([rAngle rAngle],[ang ang],'nearest','Ram-Lak',1,bigSize)/2;
if(size(I,1) > size(I,2))
        projI = irI(:, floor((size(I,1)-size(I,2))/2+2) : floor((size(I,1)-size(I,2))/2) + size(I,2)+1  );
    else
        projI = irI(floor((size(I,2)-size(I,1))/2+2) : floor((size(I,2)-size(I,1))/2) + size(I,1)+1 , : );
    end
