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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Macro for fitting the DIC PSF
%% PSF = Image 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x0,y0,width,angle,amp,offset] = DICPSF2D(PSF,1E-4);

[sizey sizex] = size(PSF);
[X,Y] = meshgrid(1:sizex,1:sizey);
fit2D = amp*((X-x0)*cos(angle) + (Y-y0)*sin(angle)).*(exp(-0.5*(X-x0).^2./(width^2)-0.5*(Y-y0).^2./(width^2))) + offset;
