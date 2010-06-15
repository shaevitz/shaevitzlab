function varargout = KymoMain(varargin)
%       GUI for Kymograph.
%       Comments displayed at the command line in response 
%       to the help command. 

% (Leave a blank line following the help.)

%  Initialization tasks

Parameters.MinConnectedComponents = 100;
Parameters.SgolayHalfWindow = 10;
Parameters.NormalHalfWindow = 8;

Metadata.Directory = 0;
Metadata.YFPFiles = 0;
Metadata.RedFiles = 0;
Metadata.DICFiles = 0;
Metadata.NumYFPFiles = 0;
Metadata.NumRedFiles = 0;
Metadata.NumDICFiles = 0;

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
  'HandleVisibility', 'callback',...
  'NextPlot', 'replacechildren',...
  'Units', 'pixels',...
  'Position', [4,44,512,512],...
  'LooseInset', [0,0,0,0]);

GUI.OutputGraph = axes(...
  'Parent', GUI.f,...
  'HandleVisibility', 'callback',...
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

GUI.ExtendROIButton = uicontrol(...
  'Visible', 'off',...
  'Parent', GUI.f,...
  'Callback', @ExtendROIButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Extend ROI',...
  'Position', [1180,420,80,20]);

GUI.ResetROIButton = uicontrol(...
  'Visible', 'off',...
  'Parent', GUI.f,...
  'Callback', @ResetROIButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Reset ROI',...
  'Position', [1180,420,80,20]);

GUI.ThresholdButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @ThresholdButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Threshold',...
  'Position', [1080,390,80,20]);

GUI.ContourButton = uicontrol(...
  'Visible', 'off',...
  'Parent', GUI.f,...
  'Callback', @ContourButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Contour',...
  'Position', [1080,360,80,20]);

GUI.MidlineButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @MidlineButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Centerline',...
  'Position', [1080,360,80,20]);

GUI.PixelMapButton = uicontrol(...
  'Parent', GUI.f,...
  'Callback', @PixelMapButton_Callback,...
  'Style', 'pushbutton',...
  'String', 'Pixel Map',...
  'Position', [1080,330,80,20]);

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
  'Position', [1080,210,80,20]);

%  Initialization tasks
xlim(GUI.InputGraph, [0,512]);
ylim(GUI.InputGraph, [0,512]);
xlim(GUI.OutputGraph, [0,512]);
ylim(GUI.OutputGraph, [0,512]);
set(GUI.InputSlider, 'Value', 1);

%  Class methods

% --- 
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

% --- 
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

% --- 
function UpdateInputGraph()
  Display.InputImage = imread(strcat(Metadata.Directory, '/', Display.Files(Display.InputIndex).name), 'TIFF');
  axes(GUI.InputGraph);
  imagesc(Display.InputImage);
  set(GUI.InputFrameCounter, 'String', strcat(num2str(Display.InputIndex), '/', num2str(Display.Num)));
  DisplayAllROIBorders();
end

% ---
function UpdateOutputGraph(this_image)
  axes(GUI.OutputGraph);
  imagesc(this_image);
end

% --- 
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
function ResetStack()
  
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
%      imagesc(Display.OutputImage);
%      drawnow;
%      pause(1/30);
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

% --- 
function ExtendROIButton_Callback(hObject, eventdata, handles)
  
end

% --- 
function ResetROIButton_Callback(hObject, eventdata, handles)
  
end

% --- Perform simple threshold on current frame
function ThresholdButton_Callback(hObject, eventdata, handles)
  ROI.Images = {};
  Display.ROI = {};
  for i = 1:ROI.N
    this_rect = ROI.Rects(1,4*i-3:4*i);
    x = round(this_rect(1));
    y = round(this_rect(2));
    w = round(this_rect(3));
    h = round(this_rect(4));
    this_image = KymoThreshold(Display.Average(y:y+h-1,x:x+w-1));
    this_image = bwareaopen(this_image, Parameters.MinConnectedComponents);
    this_image = bwmorph(this_image, 'spur');
    this_image = bwmorph(this_image, 'majority');
%    this_image = bwmorph(this_image, 'close');
    this_image = Display.Average(y:y+h-1,x:x+w-1).*double(this_image);
%    UpdateOutputGraph(this_image);
    ROI.Images = [ROI.Images this_image];
  end
  Display.ROI = [Display.ROI ROI.Images];
  UpdateField();
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
end

% --- 
function ContourButton_Callback(hObject, eventdata, handles)
  
end

% --- 
function MidlineButton_Callback(hObject, eventdata, handles)
  ROI.Contours = {};
  ROI.Retracts = {};
  ROI.Ends = {};
  Display.ROI = {};
  for i = 1:ROI.N
    this_image = cell2mat(ROI.Images(1,i));
    
    % get full retract
    [contour retract ends poles] = KymoRetract(this_image);
%    UpdateOutputGraph(max(contour, retract));
%    UpdateOutputGraph(retract);
%    for i = 1:m
%      [ext(1,i),ext(2,i)]
%      rectangle('Position', [ext(2,i),ext(1,i),1,1], 'EdgeColor', [0.5 0 0]);
%    end
%    for i = 1:2
%      rectangle('Position', [ends(i,:),1,1], 'EdgeColor', [1 i/2 0]);
%      rectangle('Position', [poles(i,:),1,1], 'EdgeColor', [0.3 i/2 0]);
%    end
%    Display.Contour = contour;
%    Display.Retract = retract;
    ROI.Contours = [ROI.Contours contour];
    ROI.Retracts = [ROI.Retracts retract];
    ROI.Ends = [ROI.Ends ends];
    outline = max(contour, retract);
    Display.ROI = [Display.ROI outline];
    
    % old
%    this_image = bwmorph(this_image, 'thin', Inf);
%    this_image = bwmorph(this_image, 'close');
%    midline = this_image;
%    endpoints = bwmorph(midline, 'endpoints');
%    % Test 1 coordinate first
%    z = size(midline); x = z(2); y = z(1);
%    % Get a coord from the endpoints
%    for i = 1:y
%      for j = 1:x
%        if endpoints(i,j) > 0
%          coords = [i,j];
%        end
%      end
%    end
%    this_image = cell2mat(ROI.Images(1,1));
%    this_image = edge(this_image);
%    contour = this_image;
%    % Find the corresponding pole on the contour
%    norms = zeros(y,x);
%    greatest_norm = Inf; g_coords = [Inf, Inf];
%    for i = 1:y
%      for j = 1:x
%        if contour(i,j) > 0
%          norms(i,j) = norm([i,j]-coords);
%          if norms(i,j) < greatest_norm;
%            greatest_norm = norms(i,j);
%            g_coords = [j,i]; % what is the correct order?
%          end
%        end
%      end
%    end
%    %g_coords
%    %this_image = contour+midline;
%    %UpdateOutputGraph(this_image);
%    %rectangle('Position', [g_coords,1,1]);
    
  end
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
end

% --- 
function PixelMapButton_Callback(hObject, eventdata, handles)
  ROI.Poles = {};
  ROI.Extends = {};
  ROI.Normals = {};
  ROI.YFPPixelMap = {};
  ROI.RedPixelMap = {};
  for i = 1:ROI.N
%    [retract_r retract_c] = find(cell2mat(ROI.Retracts(1,i))>0);
%    poles = cell2mat(ROI.Poles(1,i));
%    num_pixels = length(retract_r);
%    dx = 0.1;
%    h = 8;
%    half_box = 8;
%    [spline_x spline_y] = Spline2d([retract_c retract_r], 10);
%    figure
%    plot(spline_y, spline_x);
%    % instead use splinefit
%    try
%      xx = 1:dx:ROI.Rects(4*i-1);
%      xm = min(retract_r):dx:max(retract_r);
%      coord = 2;
%      pp = splinefit(retract_r, retract_c, 6);
%    catch exception
%      xx = 1:dx:ROI.Rects(4*i-2);
%      xm = min(retract_c):dx:max(retract_c);
%      coord = 1;
%      pp = splinefit(retract_c, retract_r, 6);
%    end
%    % find derivatives with sgolay
%    [b g] = sgolay(3, 1+2*h);
%    curve_raw = ppval(pp, xx);
%    curve_s0 = zeros(1, length(curve_raw)-2*h);
%    curve_s1 = zeros(1, length(curve_raw)-2*h);
%    curve_u = interparc(num_pixels, xx, curve_raw, 'linear');
%    for j = 1:length(curve_raw)-2*h
%      curve_s0(j) = dot(g(:,1), curve_raw(j:j+2*h));
%      curve_s1(j) = dot(g(:,2), curve_raw(j:j+2*h))/dx;
%    end
%    length(xx)
%    length(curve_s0)
%    length(curve_s1)
%    curve_uf = interparc(num_pixels, xm, curve_s0, 'linear');
%    curve_dv = interparc(num_pixels, xm, curve_s1, 'linear');
%    curve_nm = -1./curve_dv; % normal derivative
    
%    pd = polyder(pp.coefs);
%    curve_dv = ppval(mkpp(breaks, pd, d), xx);
%    % 'spline' slower than 'linear'
%    curve_u = interparc(num_pixels, xx, curve_raw, 'linear');
%    curve_dv_u = interparc(num_pixels, xx, curve_dv, 'linear');
%    % find normal lines for each point in curve_u, take mean on line
    
%    for j = 1:num_pixels
%      x = retract_c(j);
%      y = retract_r(j);
%      x_box = x-half_box:x+half_box;
%      y_box = y-half_box:y+half_box;
%      x_step = sqrt(1/(1+curve_nm(j)^2));
%      % find nearest lattice points (pixels) along normal
%      if x_step == 0 % if vertical line (dunno if this actually happens)
%        y_step = 1;
%        for k = -half_box:half_box
%          
%        end
%      else
%        for k = -ceil(1.5*half_box):ceil(1.5*half_box)
%          % for each line point in the box, add its coordinate to the cell
%          xn = x+k*x_step;
%          yn = y+normal*k*x_step;
%          [xn yn] = round([xn yn]);
%          if ROI.Images(yn,xn) > 0
%            ROI.Normals(i,j) = [ROI.Normals(i,j) [xn yn]];
%          end
%        end
%      end
%    end
    
    [normals extend poles] = KymoNormals(cell2mat(ROI.Retracts(1,i)), cell2mat(ROI.Ends(1,i)), cell2mat(ROI.Images(1,i)), Parameters.NormalHalfWindow);
    ROI.Poles = [ROI.Poles poles];
    ROI.Extends = [ROI.Extends extends];
    outline = max(cell2mat(ROI.Contours(1,i)), extends);
    Display.ROI = [Display.ROI outline];
    
    num_pixels = length(normals);
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
      this_image = imread(fullfile(Metadata.Directory, Metadata.YFPFiles(j).name), 'TIFF');
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
    title(strcat('YFP/GFP ROI', num2str(i)));
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
      this_image = imread(fullfile(Metadata.Directory, Metadata.RedFiles(j).name), 'TIFF');
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
    title(strcat('Red/mCherry ROI', num2str(i)));
    ROI.RedPixelMap = [ROI.RedPixelMap pixel_map];
  end
  UpdateOutputGraph(cell2mat(Display.ROI(1,1)));
end

% --- 
function SaveButton_Callback(hObject, eventdata, handles)
  [savefile savepath] = uiputfile();
  savefile, savepath
  save(fullfile(savepath, savefile), 'Parameters', 'Metadata', 'ROI');
end

%  Utility functions

end
