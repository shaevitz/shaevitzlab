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


function [x0,y0,width,angle,amp,offset] = DICPSF2D(PSF,tol);

% Fits a DIC PSF to a theoretical model based on a steerable 1D gaussian derivative
% [x0,y0,width,angle,amp,offset] = DICPSF2D(PSF,tol)
%
% INPUTS:
% PSF = measured PSF image
% tol = tolerance fitting angle (default ~ 1E-5)
%
% OUTPUTS:
% x0,y0 = center pixel value for the psf
% width = gaussian width (blur) of the PSF
% angle = shear angle of DIC
% amp = size of PSF signal
% offset = background intensity value

options = optimset('Display','off','TolFun',tol,'LargeScale','off','MaxFunEvals',1E5);

[sizey sizex] = size(PSF);
[X,Y] = meshgrid(1:sizex,1:sizey);


% Autogeneration of input parameters
% All based on where the min and max points are in 2D

[val, maxX, maxY] = max2d(PSF);
[val, minX, minY] = min2d(PSF);

y0 = (maxX + minX)/2;
x0 = (maxY + minY)/2;
width = sqrt( (maxX - minX)^2 + (maxY - minY)^2 ) / 2;
angle = atan2((maxY - minY),(maxX - minX)) - 1.5708;
amp = (PSF(maxX,maxY) - PSF(minX,minY))/2;
offset = (PSF(maxX,maxY) + PSF(minX,minY))/2;

initpar = [x0,y0,width,angle,amp,offset];

fp = fminunc(@fitdicpsf2D,initpar,options,PSF,X,Y);
x0 = fp(1);
y0 = fp(2);
width = fp(3);
angle = fp(4);
amp = fp(5);
offset = fp(6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = fitdicpsf2D(p,PSF,X,Y);

x0 = p(1);
y0 = p(2);
width = p(3);
angle = p(4);
amp = p(5);
offset = p(6);

ztmp = amp*((X-x0)*cos(angle) + (Y-y0)*sin(angle)).*(exp(-0.5*(X-x0).^2./(width^2)-0.5*(Y-y0).^2./(width^2))) + offset - PSF;

z = sum(sum(ztmp.^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [val, maxX, maxY] = max2d(A)

[m1,i]=max(A);
[m2,j]=max(m1);

val=m2;
maxX=i(j);
maxY=j;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [val, minX, minY] = min2d(A)

[m1,i]=min(A);
[m2,j]=min(m1);

val=m2;
minX=i(j);
minY=j;