function sIRGB=smoothScribbleImage(IYCbCr,grayImg)
% smoothing the initial colorized result IYCbCr using the grayImg as the guide

% grayYCbCr=cat(3,grayImg,grayImg,grayImg);
% grayY=grayYCbCr(:,:,1);

grayY=grayImg;


r = 4; % local window radius: r
eps = 0.02^2; % regularization parameter: eps

% smoothing the Cb Cr channels
q = zeros(size(IYCbCr));
q(:, :, 1) = IYCbCr(:,:,1);
q(:, :, 2) = guidedfilter(grayY,IYCbCr(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(grayY,IYCbCr(:, :, 3), r, eps);

sIRGB=ycbcr2rgb(q);
