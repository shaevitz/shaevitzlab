function [shear_angle] = shear(shear_img)

%   s = strel('disk', 5);
%   shear_img1 = imtophat(shear_img, s);
%   shear_angle1 = shearAngle(shear_img1);
%   shear_img2 = double(uniBackground(shear_img, 50));
%   shear_angle2 = shearAngle(shear_img2);
%   shear_angle = (shear_angle1+shear_angle2)/2;

  % Using PFDIC code again to find the shear angle
%   shear_img = double(uniBackground(shear_img, 50));
  shear_bg = DeNoise(shear_img, 20);
  shear_img = shear_img./shear_bg;
  shear_img = DeNoise(shear_img, 5);
  shear_angle = shearAngle(shear_img)+45;

end
