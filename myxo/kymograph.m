% Copyright (C) 2010, Peter Jin and Mingzhai Sun
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
% 
% Authors: 
% Peter Jin 
% peterhaijin@gmail.com
% 
% Mingzhai Sun
% mingzhai@gmail.com
% 
% v1.0 16-June-2010

function varargout = kymograph(varargin)

%  Initialization tasks

Parameters.MinConnectedComponents = 100;
Parameters.SgolayHalfWindow = 10; % TODO deprecated
Parameters.NormalHalfWindow = 8; % = 15
Parameters.Dilations = 4; % TODO deprecated
Parameters.ExtensionLength = 15;

Metadata.Directory = 0;
Metadata.YFPFiles = 0;
Metadata.RedFiles = 0;
Metadata.DICFiles = 0;
Metadata.NumYFPFiles = 0;
Metadata.NumRedFiles = 0;
Metadata.NumDICFiles = 0;
Metadata.DICStep = 0;
Metadata.DICOffset = 0;

ROI.N = 0; % number of regions of interest (N)
ROI.Rects = []; % bounding rectangle
ROI.Polys = {}; % TODO bounding polygon
ROI.Images = {}; % rectangular threshold images
ROI.Contours = {}; % binary contour
ROI.Retracts = {}; % binary retract of the threshold via thinning
ROI.Ends = {}; % head/tail inner endpoints
ROI.Poles = {}; % poles from KymoNormals
ROI.Extends = {}; % extended retracts from KymoNormals
ROI.Normals = {}; % pixel coordinates for each segment; X by N cell
ROI.YFPPixelMap = {};
ROI.RedPixelMap = {};
ROI.YFPDICMap = {};
ROI.RedDICMap = {};
ROI.YFPDICEnds = {};
ROI.RedDICEnds = {};

Display.InputIndex = 0;
Display.InputImage = 0;
Display.InputType = 'yfp';
Display.OutputIndex = 0;
Display.OutputImage = 0;
Display.Files = 0;
Display.Images = 0;
Display.Num = 0;
Display.ROI = {};
Display.Average = 0;
Display.Contour = 0;
Display.Retract = 0;

%  Construct the components

GUI.f = figure(...
  'Visible', 'on',...
  'Name', 'Kymograph',...
  'NumberTitle', 'off',...
  'Position', [100,300,1280,560],...
  'Resize', 'on',...
  'MenuBar', 'figure',...
  'Toolbar', 'figure');

GUI.InputGraph = axes(...
  'Parent', GUI.f,...
  'HandleVisibility', 'on',...
  'NextPlot', 'replacechildren',...
  'Units', 'pixels',...
  'Position', [4,44,512,512],...
  'LooseInset', [0,0,0,0]);

GUI.OutputGraph = axes(...
  'Parent', GUI.f,...
  'HandleVisibility', 'on',...
  'NextPlot', 'replacechildren',...
  'Units', 'pixels',...
  'Position', [544,44,512,512],...
  'LooseInset', [0,0,0,0]);

GUI.InputSlider = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @InputSlider_Callback,...
  'Style', 'slider',...
  'Units', 'pixels',...
  'Position', [5,6,512,18]);

GUI.OutputSlider = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @OutputSlider_Callback,...
  'Style', 'slider',...
  'Units', 'pixels',...
  'Position', [545,6,512,18]);

GUI.LoadStackButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @LoadStackButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Load Image Stack',...
  'Position', [1080,518,180,20]);

GUI.Label_StackDir = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'edit',...
  'String', '',...
  'Position', [1080,484,180,20]);

GUI.Label_ROIInput = uicontrol(...
  'Visible', 'off',...
  'Parent', GUI.f,...
  'Style', 'edit',...
  'String', '',...
  'Position', [1080,484,180,20]);

GUI.StackMenu = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @StackMenu_Callback,...
  'Style', 'popupmenu',...
  'String', {'YFP','Red','DIC'},...
  'Value', 1,...,
  'Position', [1180,450,80,20]);

GUI.InputFrameCounter = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'text',...
  'String', '',...
  'Position', [1180,420,80,16]);

GUI.AverageButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @AverageButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Average',...
  'Position', [1080,450,80,20]);

GUI.CropButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @CropButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Add ROI',...
  'Position', [1080,420,80,20]);

GUI.ThresholdButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @ThresholdButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Threshold',...
  'Position', [1080,390,80,20]);

GUI.PixelMapButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @PixelMapButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Fluo. Map',...
  'Position', [1080,360,80,20]);

GUI.DICPixelMapButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @DICPixelMapButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'DIC Map',...
  'Position', [1080,330,80,20]);

GUI.DICFrameMapButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @DICFrameMap_Callback,...
  'Style', 'pushbutton',...
  'String', 'DIC Map 2',...
  'Position', [1180,330,80,20]);

GUI.Label_YFPFrames = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'text',...
  'String', 'YFP Frames:',...
  'Position', [1080,300,120,16]);

GUI.Label_RedFrames = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'text',...
  'String', 'Red Frames:',...
  'Position', [1080,280,120,16]);

GUI.Label_DICFrames = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'text',...
  'String', 'DIC Frames:',...
  'Position', [1080,260,120,16]);

GUI.Label_ROIFields = uicontrol(...
  'Parent', GUI.f,...
  'Style', 'text',...
  'String', 'ROI Fields:',...
  'Position', [1080,240,120,16]);

GUI.SaveButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @SaveButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Save Data',...
  'Position', [1080,200,180,20]);

GUI.UndockInputButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @UndockInputButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Left Graph',...
  'Position', [1080,160,80,20]);

GUI.UndockOutputButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @UndockOutputButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Right Graph',...
  'Position', [1180,160,80,20]);

%  Initialization tasks
xlim(GUI.InputGraph, [0,512]);
ylim(GUI.InputGraph, [0,512]);
xlim(GUI.OutputGraph, [0,512]);
ylim(GUI.OutputGraph, [0,512]);
set(GUI.InputSlider, 'Value', 1);

%  Class methods

% Update the active (output-side) image stack
function UpdateStack()
  val = get(GUI.StackMenu, 'Value');
  switch val
  case 1 % YFP
    Display.Files = Metadata.YFPFiles;
    Display.Num = Metadata.NumYFPFiles(1);
  case 2 % Red
    Display.Files = Metadata.RedFiles;
    Display.Num = Metadata.NumRedFiles(1);
  case 3 % DIC
    Display.Files = Metadata.DICFiles;
    Display.Num = Metadata.NumDICFiles(1);
  end
  % Fix out-of-bounds problem with slider
  if get(GUI.InputSlider, 'Value') > Display.Num
    Display.InputIndex = Display.Num;
    set(GUI.InputSlider, 'Value', Display.InputIndex);
  elseif get(GUI.InputSlider, 'Value') < 1
    Display.InputIndex = 1;
    set(GUI.InputSlider, 'Value', Display.InputIndex);
  end
  set(GUI.InputSlider, 'Max', Display.Num);
  if Display.Num > 1
    set(GUI.InputSlider, 'Min', 1);
  else
    set(GUI.InputSlider, 'Min', 0);
  end
  set(GUI.InputSlider, 'SliderStep', [1/(Display.Num-1),10/(Display.Num-1)]);
end

% Update the output slider when the ROIs change
function UpdateField()
  Display.OutputIndex = 1;
  set(GUI.OutputSlider, 'Value', 1);
  if ROI.N > 1
    set(GUI.OutputSlider, 'SliderStep', [1/(ROI.N-1),10/(ROI.N-1)]);
    set(GUI.OutputSlider, 'Max', ROI.N);
    set(GUI.OutputSlider, 'Min', 1);
  else
    set(GUI.OutputSlider, 'SliderStep', [0,0]);
    set(GUI.OutputSlider, 'Max', 1);
    set(GUI.OutputSlider, 'Min', 0);
  end
end

% Update the input graph with the current frame of the active stack
function UpdateInputGraph()
  Display.InputImage = imread(strcat(Metadata.Directory, '/', Display.Files(Display.InputIndex).name), 'TIFF');
  axes(GUI.InputGraph);
  imagesc(Display.InputImage);
  set(GUI.InputFrameCounter, 'String', strcat(num2str(Display.InputIndex), '/', num2str(Display.Num)));
  DisplayAllROIBorders();
end

% Update the output graph with arbitrary image
function UpdateOutputGraph(this_image)
  Display.OutputImage = this_image;
  axes(GUI.OutputGraph);
  imagesc(Display.OutputImage);
end

% Display an ROI border on the active axes
function DisplayROIBorder(i)
  if i > 0 && i <= Display.Num
    this_rect = ROI.Rects(1,4*i-3:4*i);
    rectangle('Position', this_rect);
    text('Position', this_rect(1:2)-10, 'String', num2str(i));
  end
end

% --- 
function DisplayAllROIBorders()
  if ROI.N ~= 0
    for i = 1:ROI.N
      DisplayROIBorder(i);
    end
  end
end

% --- 
function ResetROI()
  ROI.N = 0;
  ROI.Rects = [];
  ROI.Polys = {};
  ROI.Images = {};
  ROI.Contours = {};
  ROI.Retracts = {};
  ROI.Ends = {};
  ROI.Poles = {};
  ROI.Extends = {};
  ROI.Normals = {};
  ROI.YFPPixelMap = {};
  ROI.RedPixelMap = {};
end

% --- 
function FluorescenceNormals()
  
end

% --- 
function [head_t tail_t] = DICNormals(j, filter, col_pixels, extend, normals, normals_ext, x, y, w, h)
  scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.DICFiles(Metadata.DICStep*(j-1)+1+Metadata.DICOffset).name), 'TIFF'));
  scaled_image = scaled_image(y:y+h-1,x:x+w-1);
  
  % Adjust the DIC contrast for a uniform threshold
%  blur = imfilter(scaled_image, filter);
%  scaled_image = scaled_image-blur;
  scaled_image = scaled_image-mean2(scaled_image);
%  scaled_image = 1000*scaled_image;
  scaled_image = 1000*mean(std(scaled_image))*scaled_image./max(max(scaled_image));
  
  % Locally close mask from DIC, then find points near poles
  scaled_image = abs(scaled_image);
  mask = threshold(scaled_image, 50);
  mask = bwmorph(mask, 'dilate', 5);
  ends = bwmorph(mask, 'thin', Inf);
  ends = bwmorph(ends, 'endpoints');
  [end_v end_u] = find(ends > 0);
  for k = 1:length(end_u)
    mask = localclose(mask, [end_u(k) end_v(k)], 15);
  end
  scaled_image = mask.*abs(scaled_image);
  
  % First approximation of the DIC poles
  [v u] = find(extend > 0);
  ends = bwmorph(extend, 'endpoints');
  [end_r end_c] = find(ends > 0);
  [u v] = eusort2(u, v, [end_c(1) end_r(1)]); % nnsort2
  [int_v int_u] = find(extend.*mask > 0);
  num_intersect = length(int_u);
  assert(num_intersect >= 2);
  t = [];
  for k = 1:num_intersect
    t = [t intersect(find(v == int_v(k)), find(u == int_u(k)))];
  end
  % these are the indices of the raw intersections with the mask
  head_t = min(t);
  tail_t = max(t);
  
  % Second approximation of the DIC poles
  if head_t > 1
    for k = head_t-1:-1:1
      line = cell2mat(normals_ext(1,k));
      this_len = 0;
      for l = 1:length(line(:,1))
        if mask(line(l,2),line(l,1)) > 0
          this_len = this_len+1;
        end
      end
      if this_len == 0
        head_t = k+1;
        break
      end
    end
  end
  if tail_t < length(u)
    for k = tail_t+1:length(u)
      line = cell2mat(normals_ext(1,k));
      this_len = 0;
      for l = 1:length(line(:,1))
        if mask(line(l,2),line(l,1)) > 0
          this_len = this_len+1;
        end
      end
      if this_len == 0
        tail_t = k-1;
        break
      end
    end
  end
  
%  figure;
%  imagesc(scaled_image);
  
%  if tail_t-head_t < 0.9*mean(col_pixels)
%    figure; plot(u,v);
%    figure;
%    imagesc(scaled_image+10*mean(std(scaled_image))*extend);
%    title(strcat(num2str(j), ' length ', num2str(tail_t-head_t+1)));
%  end
end

% --- 
function [pixel_map heads tails] = DICCCNormals(extend, normals, normals_ext, x, y, w, h, n, files)

  [v u] = find(extend > 0);
  all_pixels = length(u);
  
  heads = [];
  tails = [];
  col_pixels = [];
  
  for j = 1:n
    scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.DICFiles(Metadata.DICStep*(j-1)+1+Metadata.DICOffset).name), 'TIFF'));
    scaled_image = scaled_image(y:y+h-1,x:x+w-1);
    
    % Adjust the DIC contrast for a uniform threshold
    scaled_image = scaled_image-mean2(scaled_image);
    scaled_image = 10000*scaled_image/std(scaled_image(:));
    
    % Locally close mask from DIC, then find points near poles
    scaled_image = abs(scaled_image);
    mask = threshold(scaled_image, 50);
    mask = bwmorph(mask, 'dilate', 5);
    ends = bwmorph(mask, 'thin', Inf);
    ends = bwmorph(ends, 'endpoints');
    [end_v end_u] = find(ends > 0);
    for k = 1:length(end_u)
      mask = localclose(mask, [end_u(k) end_v(k)], 15);
    end
    scaled_image = mask.*abs(scaled_image);
    
    % First approximation of the DIC poles
    [v u] = find(extend > 0);
    ends = bwmorph(extend, 'endpoints');
    [end_r end_c] = find(ends > 0);
    [u v] = eusort2(u, v, [end_c(1) end_r(1)]); % nnsort2
    [int_v int_u] = find(extend.*mask > 0);
    num_intersect = length(int_u);
    assert(num_intersect >= 2);
    t = [];
    for k = 1:num_intersect
      t = [t intersect(find(v == int_v(k)), find(u == int_u(k)))];
    end
    % these are the indices of the raw intersections with the mask
    head_t = min(t);
    tail_t = max(t);num_intersect = length(int_u);
    assert(num_intersect >= 2);
    t = [];
    for k = 1:num_intersect
      t = [t intersect(find(v == int_v(k)), find(u == int_u(k)))];
    end
    % these are the indices of the raw intersections with the mask
    head_t = min(t);
    tail_t = max(t);
    
    % Second approximation of the DIC poles
%    if head_t > 1
%      for k = head_t-1:-1:1
%        line = cell2mat(normals_ext(1,k));
%        this_len = 0;
%        for l = 1:length(line(:,1))
%          if mask(line(l,2),line(l,1)) > 0
%            this_len = this_len+1;
%          end
%        end
%        if this_len == 0
%          head_t = k+1;
%          break
%        end
%      end
%    end
%    if tail_t < length(u)
%      for k = tail_t+1:length(u)
%        line = cell2mat(normals_ext(1,k));
%        this_len = 0;
%        for l = 1:length(line(:,1))
%          if mask(line(l,2),line(l,1)) > 0
%            this_len = this_len+1;
%          end
%        end
%        if this_len == 0
%          tail_t = k-1;
%          break
%        end
%      end
%    end
    
    % A different second approximation: template matching.
    template = double(imread('data/circle10.png', 'PNG'));
    template = template/max(template(:));
    intersection = extend.*edge(filter2(template, scaled_image), 'canny');
    [corr_v corr_u] = find(intersection > 0);
    num_intersect = length(corr_u);
    if num_intersect >= 2
      t = [];
      for k = 1:num_intersect
        t = [t intersect(find(v == corr_v(k)), find(u == corr_u(k)))];
      end
%      if head_t > min(t)-10
%        head_t = max(1, min(t)-8);
%      end
%      if tail_t < max(t)+10
%        tail_t = min(all_pixels, max(t)+8);
%      end
      if head_t > min(t)
        head_t = max(1, min(t)-6);
      end
      if tail_t < max(t)
        tail_t = min(all_pixels, max(t)+6);
      end
    end
    
    num_pixels = tail_t-head_t+1;
    col_pixels = [col_pixels num_pixels];
    heads = [heads head_t];
    tails = [tails tail_t];
    
    % Compute the pixel map
    scaled_image = double(imread(fullfile(Metadata.Directory, files(j).name), 'TIFF'));
    scaled_image = scaled_image(y:y+h-1,x:x+w-1);
    pixel_col = zeros(all_pixels, 1);
    for k = head_t:tail_t%1:num_pixels
      line = cell2mat(normals(1,k));
      line_pixels = impixel(scaled_image, line(:,1), line(:,2));
      pixel_col(k) = mean(line_pixels(:,1));
    end
    if j > 1
      map_size = size(pixel_map);
      min_length = min(map_size(1), length(pixel_col));
      pixel_map = [pixel_map(1:min_length,:) pixel_col(1:min_length,:)];
    else
      pixel_map = [pixel_col];
    end
%    if ceil(j/5) == floor(j/5)
%      fprintf(1, 'Status: %f%%\n', 100*j/n);
%    end
  end
%  pm = pixel_map;
  target_length = min(tails-heads)-5;
  [pixel_map heads tails] = MapAlign(pixel_map, target_length, heads, tails, n);
%  for j = 1:n
%    pm(heads(j):tails(j),j) = 5*pm(heads(j):tails(j),j);
%  end
%  figure; imagesc(pm);
end

% --- 
function [new_pixel_map heads tails] = MapAlign(pixel_map, target_length, heads, tails, n)
  % jitter correction by cross correlation
  if tails(1)-heads(1)+1 > target_length
    len_diff = tails(1)-heads(1)+1-target_length;
    head_corr = round(len_diff/2);
    tail_corr = len_diff-head_corr;
    heads(1) = heads(1)+head_corr;
    tails(1) = tails(1)-tail_corr;
  end
  new_pixel_map = pixel_map(heads(1):tails(1),1);
  for j = 2:n
    this_length = tails(j)-heads(j)+1;
    len_diff = this_length-target_length;
    if len_diff > 0
      ldiffs = [];
      rdiffs = [];
      last_col = new_pixel_map(:,j-1);
      last_col = (last_col-mean(last_col))/std(last_col);
      for k = 0:len_diff
        this_col = pixel_map(heads(j)+k:heads(j)+k+target_length-1,j);
        this_col = (this_col-mean(this_col))/std(this_col);
        ldiff = (this_col.*last_col);
        ldiffs = [ldiffs sum(ldiff)];
      end
%      for k = 0:len_diff
%        this_col = pixel_map(tails(j)-k:-1:tails(j)-k-target_length+1,j);
%        this_col = (this_col-mean(this_col))/std(this_col);
%        rdiff = (this_col.*last_col);
%        rdiffs = [rdiffs sum(rdiff)];
%      end
      ltop = find(ldiffs == max(ldiffs));
%      rtop = find(rdiffs == max(rdiffs));
%      if max(ldiffs) >= max(rdiffs) || rtop == NaN
        l_head = heads(j)+round(mean(ltop))-1;
        l_tail = l_head+target_length-1;
%      else
%        r_tail = tails(j)-rtop+1;
%        r_head = r_tail-target_length+1;
%      end
    end
%    if max(ldiffs) >= max(rdiffs)
%      [j l_head l_tail]
      new_pixel_map = [new_pixel_map pixel_map(l_head:l_tail,j)];
      heads(j) = l_head;
      tails(j) = l_tail;
%    else
%      [j r_head r_tail]
%      this_col = pixel_map(r_head:r_tail,j);
%      new_pixel_map = [new_pixel_map this_col(end:-1:1)];
%      heads(j) = r_head;
%      tails(j) = r_tail;
%    end
  end
end

% --- 
function [offset] = LinearAlign(old, new)
  if length(new) > length(old)
    static = old;
    moving = new;
  else
    static = new;
    moving = old;
  end
  ldiffs = [];
  last_col = (static-mean(static))/std(static);
  this_col = (moving-mean(moving))/std(moving);
  min_length = min(length(static), length(moving));
  for i = 1:length(moving)-length(static)+1
    ldiff = this_col(i:min_length-1).*last_col;
    ldiffs = [ldiffs ldiff];
  end
  top = find(ldiffs == max(ldiffs));
  if length(new) > length(old)
    offset = top-1;
  else
    offset = 1-top;
  end
end

%  Callbacks

% --- 
function LoadStackButton_Callback(hObject, eventdata, handles)
  Metadata.Directory = uigetdir({}, 'Load Image Stack Directory...');
  if Metadata.Directory ~= 0
    
    Metadata.YFPFiles = dir(strcat(Metadata.Directory, '/*_YFP.tif*'));
    Metadata.RedFiles = dir(strcat(Metadata.Directory, '/*_Red.tif*'));
    Metadata.DICFiles = dir(strcat(Metadata.Directory, '/*_DIC.tif*'));
    Metadata.NumYFPFiles = length(Metadata.YFPFiles);
    Metadata.NumRedFiles = length(Metadata.RedFiles);
    Metadata.NumDICFiles = length(Metadata.DICFiles);
    Metadata.DICStep = round(Metadata.NumDICFiles/Metadata.NumYFPFiles);
    
    Metadata.DICOffset = 0;
    
    Display.InputIndex = 1;
    set(GUI.Label_StackDir, 'String', Metadata.Directory);
    set(GUI.Label_YFPFrames, 'String', strcat('YFP Frames:', num2str(Metadata.NumYFPFiles)));
    set(GUI.Label_RedFrames, 'String', strcat('Red Frames:', num2str(Metadata.NumRedFiles)));
    set(GUI.Label_DICFrames, 'String', strcat('DIC Frames:', num2str(Metadata.NumDICFiles)));
    
    UpdateStack();
    UpdateInputGraph();
  end
end

% --- 
function StackMenu_Callback(hObject, eventdata, handles)
  if Metadata.Directory ~= 0
    UpdateStack();
    UpdateInputGraph();
  end
end

% --- 
function InputSlider_Callback(hObject, eventdata, handles)
  Display.InputIndex = round(get(hObject, 'Value'));
  UpdateInputGraph();
end

% --- 
function OutputSlider_Callback(hObject, eventdata, handles)
  if ROI.N ~= 0
    Display.OutputIndex = round(get(hObject, 'Value'));
    UpdateOutputGraph(cell2mat(Display.ROI(1,Display.OutputIndex)));
  end
end

% --- 
function AverageButton_Callback(hObject, eventdata, handles)
  % Clear existing ROI fields
  ResetROI();
  Display.Average = 0;
  Display.Contour = 0;
  Display.Retract = 0;
  Display.ROI = {};
  if Display.Num > 0
    Display.OutputImage = double(imread(fullfile(Metadata.Directory, Display.Files(1).name), 'TIFF'));
    for i = 2:Display.Num
      Display.OutputImage = Display.OutputImage+double(imread(fullfile(Metadata.Directory, Display.Files(i).name), 'TIFF'));
    end
    Display.Average = Display.OutputImage/Display.Num;
    Display.ROI = [Display.ROI Display.Average];
    UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
  end
  UpdateField();
end

% --- 
function CropButton_Callback(hObject, eventdata, handles)
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
  DisplayAllROIBorders();
  this_rect = getrect(GUI.OutputGraph);
  if this_rect ~= 0
    ROI.Rects = [ROI.Rects this_rect];
    ROI.N = ROI.N+1;
    UpdateField();
    DisplayROIBorder(ROI.N);
  end
  set(GUI.Label_ROIFields, 'String', strcat('ROI Fields:', num2str(ROI.N)));
  UpdateInputGraph();
end

% --- Perform simple threshold on current frame
function ThresholdButton_Callback(hObject, eventdata, handles)
  ROI.Images = {};
  Display.ROI = {};
  Display.Contour = 0;
  Display.Retract = 0;
  % redo average; modularize
  val = get(GUI.StackMenu, 'Value');
  switch val
  case 1
    stack_files = Metadata.YFPFiles;
    num_files = Metadata.NumYFPFiles;
  case 2
    stack_files = Metadata.RedFiles;
    num_files = Metadata.NumRedFiles;
  end
  Display.OutputImage = double(imread(fullfile(Metadata.Directory, stack_files(1).name), 'TIFF'));
  for i = 2:num_files %Display.Num
    Display.OutputImage = Display.OutputImage+double(imread(fullfile(Metadata.Directory, stack_files(i).name), 'TIFF'));
  end
  Display.Average = Display.OutputImage/num_files; %Display.Num;
  mask = threshold(Display.Average, 100);
  for i = 1:ROI.N
    this_rect = ROI.Rects(1,4*i-3:4*i);
    x = round(this_rect(1));
    y = round(this_rect(2));
    w = round(this_rect(3));
    h = round(this_rect(4));
    this_mask = mask(y:y+h-1,x:x+w-1).*bwareaopen(mask(y:y+h-1,x:x+w-1), 400);
    this_image = Display.Average(y:y+h-1,x:x+w-1).*this_mask;
%    this_image = threshold(Display.Average(y:y+h-1,x:x+w-1), 100);
%    this_image = double(this_image.*Display.Average(y:y+h-1,x:x+w-1));
    ROI.Images = [ROI.Images this_image];
  end
  Display.ROI = [Display.ROI ROI.Images];
  UpdateField();
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
end

% --- 
function PixelMapButton_Callback(hObject, eventdata, handles)
  ROI.Contours = {};
  ROI.Retracts = {};
  ROI.Ends = {};
  ROI.Poles = {};
  ROI.Extends = {};
  ROI.Normals = {};
  ROI.NumPixels = []; % only initialized here
  ROI.YFPPixelMap = {};
  ROI.RedPixelMap = {};
  Display.ROI = {};
  for i = 1:ROI.N
    [contour retract ends] = KymoRetract(cell2mat(ROI.Images(1,i)));
    ROI.Contours = [ROI.Contours contour];
    ROI.Retracts = [ROI.Retracts retract];
    ROI.Ends = [ROI.Ends ends];
    
    [normals extend poles] = KymoNormals(cell2mat(ROI.Retracts(1,i)), cell2mat(ROI.Ends(1,i)), cell2mat(ROI.Images(1,i)), 15, 0, 0);
    ROI.Poles = [ROI.Poles poles];
    ROI.Extends = [ROI.Extends extend];
    
    outline = max(cell2mat(ROI.Contours(1,i)), extend);
    Display.ROI = [Display.ROI outline];
    
    num_pixels = length(normals);
    ROI.NumPixels = [ROI.NumPixels num_pixels];
    x = round(ROI.Rects(4*i-3));
    y = round(ROI.Rects(4*i-2));
    w = round(ROI.Rects(4*i-1));
    h = round(ROI.Rects(4*i));
    
%    pixel_map = [];
%    for j = 1:Metadata.NumYFPFiles
%      this_image = imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF');
%      pixel_col = impixel(this_image(y:y+h-1,x:x+w-1), retract_c, retract_r);
%      [bad_r bad_c] = find(pixel_col>2500);
%      pixel_col(bad_r,1) = mean(pixel_col(:,1));
%      pixel_map = [pixel_map pixel_col(:,1)];
%    end
%    figure
%    imagesc(pixel_map);
%    title(strcat('YFP/GFP ROI', num2str(i)));
    
    pixel_map = [];
    for j = 1:Metadata.NumYFPFiles
      this_image = double(imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF'));
      pixel_col = zeros(num_pixels, 1);
      for k = 1:num_pixels
        line = cell2mat(normals(1,k));
        these_pixels = impixel(this_image(y:y+h-1,x:x+w-1), line(:,1), line(:,2));
        pixel_col(k) = mean(these_pixels(:,1));
      end
      pixel_map = [pixel_map pixel_col];
    end
    figure
    imagesc(pixel_map);
    title(strcat('YFP/GFP Fluorescence ROI', num2str(i)));
    ROI.YFPPixelMap = [ROI.YFPPixelMap pixel_map];
    
%    pixel_map = [];
%    for j = 1:Metadata.NumRedFiles
%      this_image = imread(fullfile(Metadata.Directory, Metadata.RedFiles(j).name), 'TIFF');
%      pixel_col = impixel(this_image(y:y+h-1,x:x+w-1), retract_c, retract_r);
%      [bad_r bad_c] = find(pixel_col>2500);
%      pixel_col(bad_r,1) = mean(pixel_col(:,1));
%      pixel_map = [pixel_map pixel_col(:,1)];
%    end
%    figure
%    imagesc(pixel_map);
%    title(strcat('Red/mCherry ROI', num2str(i)));
    
    pixel_map = [];
    for j = 1:Metadata.NumRedFiles
      this_image = double(imread(fullfile(Metadata.Directory, Metadata.RedFiles(j).name), 'TIFF'));
      pixel_col = zeros(num_pixels, 1);
      for k = 1:num_pixels
        line = cell2mat(normals(1,k));
        these_pixels = impixel(this_image(y:y+h-1,x:x+w-1), line(:,1), line(:,2));
        pixel_col(k) = mean(these_pixels(:,1));
      end
      pixel_map = [pixel_map pixel_col];
    end
    figure
    imagesc(pixel_map);
    title(strcat('Red/mCherry Fluorescence ROI', num2str(i)));
    ROI.RedPixelMap = [ROI.RedPixelMap pixel_map];
  end
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
end

% --- 
function DICPixelMapButton_Callback(hObject, eventdata, handles)
%  ROI.Contours = {};
%  ROI.Retracts = {};
%  ROI.Ends = {};
%  ROI.Poles = {};
%  ROI.Extends = {};
%  ROI.Normals = {};
  ROI.YFPDICMap = {};
  ROI.RedDICMap = {};
  ROI.YFPDICEnds = {};
  ROI.RedDICEnds = {};
  for i = 1:ROI.N
    x = round(ROI.Rects(4*i-3));
    y = round(ROI.Rects(4*i-2));
    w = round(ROI.Rects(4*i-1));
    h = round(ROI.Rects(4*i));
    
    fprintf(1, 'Starting DIC pixel map...\n');
    fprintf(1, 'Runtime estimate: %f s\n', round(sqrt(w^2+h^2)/225*Metadata.NumYFPFiles));
    
    [contour retract ends] = KymoRetract(cell2mat(ROI.Images(1,i)));
    [normals extend poles] = KymoNormals(retract, ends, cell2mat(ROI.Images(1,i)), 15, 25, 0);
    [normals_ext extra1 extra2] = KymoNormals(retract, ends, ones(h,w), 15, 25, 25);
    
    [v u] = find(extend > 0);
    all_pixels = length(u);
    
    % TODO show ROI bounding box motion (impoly)
    
    pixel_map = [];
%    col_pixels = [];
%    centers = [];
%    masks = {};
    
%    filter = fspecial('gaussian', 40, 40);
    
%    tic;
%    heads = [];
%    tails = [];
%    for j = 1:Metadata.NumYFPFiles
%      [head_t tail_t] = DICNormals(j, filter, col_pixels, extend, normals, normals_ext, x, y, w, h);
%      num_pixels = tail_t-head_t+1;
%      col_pixels = [col_pixels num_pixels];
%      heads = [heads head_t];
%      tails = [tails tail_t];
%      
%      scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF'));
%      scaled_image = scaled_image(y:y+h-1,x:x+w-1);
%      pixel_col = 600*ones(all_pixels, 1);
%      for k = head_t:tail_t%1:num_pixels
%        line = cell2mat(normals(1,k));
%        line_pixels = impixel(scaled_image, line(:,1), line(:,2));
%        pixel_col(k) = mean(line_pixels(:,1));
%      end
%      
%      if j > 1
%        map_size = size(pixel_map);
%        min_length = min(map_size(1), length(pixel_col));
%        pixel_map = [pixel_map(1:min_length,:) pixel_col(1:min_length,:)];
%      else
%        pixel_map = [pixel_col];
%      end
%      if ceil(j/5) == floor(j/5)
%        fprintf(1, 'Status: %f%%\n', 100*j/Metadata.NumYFPFiles);
%      end
%%      scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF'));
%%      pixel_col = zeros(all_pixels, 1);
%%      for k = 1:all_pixels
%%        line = cell2mat(normals(1,k));
%%        line_pixels = impixel(scaled_image(y:y+h-1,x:x+w-1), line(:,1), line(:,2));
%%        pixel_col(k) = mean(line_pixels(:,1));
%%      end
%%      pixel_map = [pixel_map pixel_col];
%    end
%    col_pixels, min(col_pixels)
%    toc;
%    fprintf(1, 'Minimal column: %d\n', min(col_pixels));
    
    [pixel_map heads tails] = DICCCNormals(extend, normals, normals_ext, x, y, w, h, Metadata.NumYFPFiles, Metadata.YFPFiles);
%    pixel_map = JitterCorrect(pixel_map, heads, tails, Metadata.NumYFPFiles);
    
%    centers = round(centers);
%    width = mean(num_pixels);
%    halfw = floor(width/2);
%    pixel_map = [];
%    for j = 1:Metadata.NumYFPFiles
%      pixel_map = [pixel_map pixel_map(centers(j)-halfw:centers(j)+halfw,j)];
%    end
    
    figure;
    imagesc(pixel_map); %(max_head_t:min_tail_t,:)
    title(strcat('YFP/GFP DIC ROI', num2str(i)));
    ROI.YFPDICMap = [ROI.YFPDICMap pixel_map];
    ROI.YFPDICEnds = [ROI.YFPDICEnds [heads; tails]];
    
    pixel_map = [];
%    col_pixels = [];
%    centers = [];
    
%    tic;
%    max_head_t = 0;
%    min_tail_t = all_pixels+1;
%    for j = 1:Metadata.NumRedFiles
%      [head_t tail_t] = DICNormals(j, filter, col_pixels, extend, normals, normals_ext, x, y, w, h);
%      num_pixels = tail_t-head_t+1;
%      col_pixels = [col_pixels num_pixels];
%%      centers = [centers center];
%      heads = [heads head_t];
%      tails = [tails tail_t];
%      
%%      pixel_col = [];
%      scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.RedFiles(j).name), 'TIFF'));
%      scaled_image = scaled_image(y:y+h-1,x:x+w-1);
%      pixel_col = 600*ones(all_pixels, 1);
%      for k = head_t:tail_t%1:num_pixels
%        line = cell2mat(normals(1,k));
%        line_pixels = impixel(scaled_image, line(:,1), line(:,2));
%        pixel_col(k) = mean(line_pixels(:,1));
%      end
%      
%      if j > 1
%        map_size = size(pixel_map);
%        min_length = min(map_size(1), length(pixel_col));
%        pixel_map = [pixel_map(1:min_length,:) pixel_col(1:min_length,:)];
%      else
%        pixel_map = [pixel_col];
%      end
%      
%      if ceil(j/5) == floor(j/5)
%        fprintf(1, 'Status: %f%%\n', 100*j/Metadata.NumRedFiles);
%      end
%%      scaled_image = double(imread(fullfile(Metadata.Directory, Metadata.RedFiles(j).name), 'TIFF'));
%%      pixel_col = zeros(all_pixels, 1);
%%      for k = 1:all_pixels
%%        line = cell2mat(normals(1,k));
%%        line_pixels = impixel(scaled_image(y:y+h-1,x:x+w-1), line(:,1), line(:,2));
%%        pixel_col(k) = mean(line_pixels(:,1));
%%      end
%%      pixel_map = [pixel_map pixel_col];
%    end
%%    col_pixels, min(col_pixels)
%    toc;
%    fprintf(1, 'Minimal column: %d\n', min(col_pixels));
    
    [pixel_map heads tails] = DICCCNormals(extend, normals, normals_ext, x, y, w, h, Metadata.NumRedFiles, Metadata.RedFiles);
%    pixel_map = JitterCorrect(pixel_map, heads, tails, Metadata.NumRedFiles);
    
%    centers = round(centers);
%    width = mean(num_pixels);
%    halfw = floor(width/2);
%    pixel_map = [];
%    for j = 1:Metadata.NumRedFiles
%      pixel_map = [pixel_map pixel_map(centers(j)-halfw:centers(j)+halfw,j)];
%    end
    
    figure;
    imagesc(pixel_map); %(max_head_t:min_tail_t,:)
    title(strcat('Red/mCherry DIC ROI', num2str(i)));
    ROI.RedDICMap = [ROI.RedDICMap pixel_map];
    ROI.RedDICEnds = [ROI.RedDICEnds [heads; tails]];
    
%    [savefile savepath] = uiputfile();
%    savefile, savepath
%    save(fullfile(savepath, savefile), 'masks');
  end
end

% --- 
function DICFrameMap_Callback(hObject, eventdata, handles)
  % Find framewise retracts of a cell using correlation and edge detection
  for i = 1:ROI.N
    
    x = round(ROI.Rects(4*i-3));
    y = round(ROI.Rects(4*i-2));
    w = round(ROI.Rects(4*i-1));
    h = round(ROI.Rects(4*i));
    
    circle10 = double(imread('data/circle10.png', 'PNG'));
    circle10 = circle10/max(circle10(:));
    
    pixel_map = [];
    lengths = [];
    heads = [];
    tails = [];
    
    fprintf(1, 'Starting DIC framewise map...\n');
    tic;
    
    for j = 1:Metadata.NumYFPFiles
      
      full_image = double(imread(fullfile(Metadata.Directory, Metadata.DICFiles(Metadata.DICStep*(j-1)+1).name), 'TIFF'));
      scaled_image = full_image(y:y+h-1,x:x+w-1);
      scaled_image = scaled_image-mean2(scaled_image);
      scaled_image = abs(filter2(circle10, scaled_image));
      scaled_image = scaled_image.^2;
      
      mask = edge(scaled_image, 'sobel')+edge(scaled_image, 'log');
      mask = bwareaopen(mask > 0, 20);
      
      % If it doesn't work the first time, throw more convex hulls at it.
%      contour = edge(mask, 'canny');
      [v u] = find(mask > 0);
      t = convhull(u, v);
      for k = 1:length(t)-1
        [x1 x2 x3 vv uu] = bresenham(mask, [u(t(k)) v(t(k)); u(t(k+1)) v(t(k+1))], 0);
        for l = 1:length(uu)
          mask(vv(l),uu(l)) = 1;
        end
      end
      [x1 x2 x3 vv uu] = bresenham(mask, [u(t(1)) v(t(1)); u(t(end)) v(t(end))], 0);
      for l = 1:length(uu)
        mask(vv(l),uu(l)) = 1;
      end
%      mask = imfill(mask, 'holes');
      
%      ends = bwmorph(mask, 'endpoints');
%      [v u] = find(ends > 0);
%      for k = 1:length(u)
%        mask = localclose(mask, [u(k) v(k)], 7);
%      end
      mask = imfill(mask, 'holes');
      mask = bwareaopen(mask, 100);
      
%      ends = bwmorph(mask, 'endpoints');
%      [v u] = find(ends > 0);
%      for k = 1:length(u)
%        mask = localclose(mask, [u(k) v(k)], 30);
%      end
%      mask = bwmorph(mask, 'close');
      
      mask = bwareaopen(mask, 10);
%      mask = bwmorph(mask, 'majority');
      pad = 10;
      mask = padarray(mask, [pad pad]);
      se = strel('disk', 3);
      mask = imclose(mask, se);
%      se = strel('disk', 2);
%      mask = imclose(mask, se);
%      mask = mask(1+pad:end-pad,1+pad:end-pad);
      mask = imfill(mask, 'holes');
%      mask = bwmorph(mask, 'erode', 2);
      
%      figure;
%      imagesc(mask);
      
      retract = bwmorph(mask, 'thin', Inf);
      ends = bwmorph(retract, 'endpoints');
      [v u] = find(ends > 0);
      for k = 1:length(u)
        retract = localclose(retract, [u(k) v(k)], 15);
      end
%      retract = bwmorph(retract, 'thin', Inf);
%      retract = bwmorph(retract, 'spur');
      se = strel('disk', 5);
      retract = imclose(retract, se);
      retract = bwmorph(retract, 'thin', Inf);
      retract = retract(1+pad:end-pad,1+pad:end-pad);
      ends = bwmorph(retract, 'endpoints');
      [v u] = find(ends > 0);
      
      if length(u) ~= 2
        [u v]
        figure;
        imagesc(mask);
        figure;
        imagesc(retract);
      end
      assert(2 <= length(u));
      
      [normals extend poles] = KymoNormals(retract, [u v], mask, 15, 25, 0);
      
      full_image = double(imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF'));
      scaled_image = full_image(y:y+h-1,x:x+w-1);
      
      num_pixels = length(normals);
%      pixel_col = zeros(num_pixels, 1);
      pixel_col = zeros(w+h, 1);
      for k = 1:num_pixels
        line = cell2mat(normals(1,k));
        pixels = impixel(scaled_image, line(:,1), line(:,2));
        pixel_col(k) = mean(pixels(:,1));
      end
      lengths = [lengths num_pixels];
      min_length = min(lengths);
      heads = [heads 1];
      tails = [tails num_pixels];
%      if j > 1
        
%      end
      pixel_map = [pixel_map pixel_col];
      
      if (j/5 == round(j/5))
        fprintf(1, 'Completed: %f%%...\n', 100*j/Metadata.NumYFPFiles);
      end
      
    end
    
    [pixel_map heads tails] = MapAlign(pixel_map, min_length, heads, tails, Metadata.NumYFPFiles);
    
    fprintf(1, 'Done.\n');
    toc;
    
    figure;
    imagesc(pixel_map);
    
  end
end

% --- 
function SaveButton_Callback(hObject, eventdata, handles)
  [savefile savepath] = uiputfile();
  savefile, savepath
  save(fullfile(savepath, savefile), 'Parameters', 'Metadata', 'ROI');
  % TODO output readme file with .mat
  readmefile = strcat(savefile, '_readme');
  readme = fopen(fullfile(savepath, readmefile), 'w+');
  fprintf(readme, 'Date: %s\n', datestr(clock()));
  fprintf(readme, 'Output: %s\n', fullfile(savepath, savefile));
  fprintf(readme, 'Directory: %s\n', Metadata.Directory);
  for i = 1:ROI.N
    x = round(ROI.Rects(4*i-3));
    y = round(ROI.Rects(4*i-2));
    w = round(ROI.Rects(4*i-1));
    h = round(ROI.Rects(4*i));
    fprintf(readme, 'ROI %d: [%d %d %d %d]\n', i, x, y, w, h);
  end
  fclose(readme);
end

% --- 
function UndockInputButton_Callback(hObject, eventdata, handles)
  figure;
  imagesc(Display.InputImage);
end

% --- 
function UndockOutputButton_Callback(hObject, eventdata, handles)
  figure;
  imagesc(Display.OutputImage);
end

end % kymograph
