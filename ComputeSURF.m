function [img SURF] = ComputeSURF(imgName, ratio)
%% Convert the image into Lab Color Space.
img = im2double(imread(imgName));
img = imresize(img, ratio);

[r c d] = size(img);
if d == 3
    img = rgb2lab(img);
else
    img = cat(3, img, img, img);
    img = rgb2lab(img);
end

%% Compute the SURF features.
bUpright = true;
bExtended = true;
verbose = false;
% Create Integral Image
iimg=IntegralImage_IntegralImage(img(:, :, 1));
[row col] = size(img(:, :, 1));
SURF = zeros(row, col, 128);
for I=1:row
    clc; disp([num2str(I) '/' num2str(row)]);
    for J=1:col
        ip.x = J;
        ip.y = I;
        ip.scale = 1;
        SURF(I, J, :) =SurfDescriptor_GetDescriptor(ip, bUpright, bExtended, iimg, verbose);
    end
end
return;
function pic=IntegralImage_IntegralImage(I)
% Convert Image to double
switch(class(I));
    case 'uint8'
        I=double(I)/255;
    case 'uint16'
        I=double(I)/65535;
    case 'int8'
        I=(double(I)+128)/255;
    case 'int16'
        I=(double(I)+32768)/65535;
    otherwise
        I=double(I);
end

% Convert Image to greyscale
if(size(I,3)==3)
	cR = .2989; cG = .5870; cB = .1140;
	I=I(:,:,1)*cR+I(:,:,2)*cG+I(:,:,3)*cB;
end

% Make the integral image
pic = cumsum(cumsum(I,1),2);