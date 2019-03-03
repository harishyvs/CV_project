function Colorization(keyWord, InputImg, inpColImg, param)

%close all;
%warning off;
SelectedNN = param.nnSize;
outputDir = ['./Results/'];
dataFileName = [outputDir keyWord '_ColorizationData.mat'];
outputFileName = [outputDir num2str(SelectedNN) '_' keyWord ];

generateColorizationData = param.RecomputationRequired;

if generateColorizationData == 1
    ratio = param.inRatio;          % Image Resize Ratio.
    ratio1 = param.colRatio;        % Image Resize Ratio.    
    nnSize = 3;                     % SP NN Size.
    stdSize = 5;                    % Standard Deviation NN Size
    spCount = param.spCount;        % Number of SP to be computed for each Image.

    %% Input Gray Image 
    % Compute mean value of SURF, Texture, Intensity and SD for each SP and
    % mean value of Intensity and SD of NN SPs.
    
    % Compute the SURF features .keyWord
    clc;disp('Computing SURF for Input Image...');
    [inImg , Surf_1] = ComputeSURF(InputImg, ratio);
    % Compute the Turbo Pixels.
    clc;disp('Computing Superpixel for Input Image...');
    [label1,~] = superpixels(lab2rgb(inImg), spCount);

    inImg = inImg(:, :, 1);
    % Compute the Texture Features.
    clc;disp('Computing Texture Features for Input Image...');
    texture1 = GetTextureFeatures(inImg/100);
    % Compute the Standard Deviation.
    grayValue = inImg / 100;
    inpStd = stdfilt(grayValue, ones(stdSize));
    % Get the List of NN List.
    nnList1 = GetListofNeighbouringSuperPixels(label1, nnSize);
    
    TotalSuperpixels = max(label1(:));    
    fcount = size(Surf_1, 3);
    spStd1 = zeros(TotalSuperpixels, 4);
    spSIFT1 = zeros(TotalSuperpixels, fcount);
    spTexture1 = zeros(TotalSuperpixels, size(texture1, 3));
    ImgRow = size(inImg,1);     ImgCol = size(inImg,2);
    Surf_1 = reshape(Surf_1, [ImgRow*ImgCol fcount]);
    texture1 = reshape(texture1, [ImgRow*ImgCol size(texture1,3)]); 
    
    for I = 1:TotalSuperpixels
        clc;disp(['Processing Image 1: '  num2str(I) ' of ' num2str(TotalSuperpixels)]);
        sp = (label1 == I);
        currSP_Idx = sp==1;
        currSP_sift = Surf_1(currSP_Idx,:);
        spSIFT1(I,:) = mean(currSP_sift, 1);              

        % Compute the mean SIFT Value of each Super Pixel.
%         for J = 1:fcount
%             p = Surf_1(:,:,J);
%             spSIFT1(I, J) = mean(p(sp));        
%         end
        % Compute the mean Texture Value of each Super Pixel.
%         for J = 1:size(texture1, 3)
%             p = texture1(:,:,J);
%             spTexture1(I, J) = mean(p(sp));        
%         end 
        currSP_texture = texture1(currSP_Idx,:);
        spTexture1(I,:) = mean(currSP_texture, 1); 
        % Compute the mean intensity value of each Super Pixel.
        spStd1(I, 1) = mean(grayValue(sp));
        spStd1(I, 2) = 0;
        % Compute the mean Standard Deviation value of each Super Pixel.
        spStd1(I, 3) = mean(inpStd(sp));
        spStd1(I, 4) = 0;
        % Compute the mean Intensity and SD value of NN Super Pixels.
        for J = 1:nnList1{I, 2}
            spNN = nnList1{I, 3};
            spStd1(I, 2) = spStd1(I, 2) + mean(grayValue((label1 == spNN(J))));
            spStd1(I, 4) = spStd1(I, 4) + mean(inpStd((label1 == spNN(J))));
        end 
        spStd1(I, 2) = spStd1(I, 2) / nnList1{I, 2};
        spStd1(I, 4) = spStd1(I, 4) / nnList1{I, 2};
    end    
    
    %% Input Color Images
    % For each Input color Image, compute mean value of SIFT, Texture, 
    % Intensity and SD for each SP and mean value of Intensity and SD 
    % of NN SPs and SP Centroid's Color Value (a & b).

    % Each row will represent an Image and each column will contain:
    % 1st Column    : SIFT Value.
    % 2nd Column    : Texture Value.
    % 3rd Column    : Intensity and Texture Values.
    % 4th Column    : Centroid color value (a & b channel).
    colImgFeatureCell = cell(size(inpColImg, 1), 5);
    TotalColorSuperPixels = 0;
    for counter = 1:size(inpColImg, 1)
        % Compute the SURF features.
        clc;disp(['Computing SURF for Color Image...' num2str(counter)]);
        [inCol Sift2] = ComputeSURF(inpColImg{counter}, ratio1); 
        
        % Compute the Turbo Pixels.
        clc;disp(['Computing Super Pixel for Color Image...' num2str(counter)]);
        [label2,~] = superpixels(lab2rgb(inCol), spCount);

        % Compute the Texture Features.
        clc;disp(['Computing Texture Features for Color Image...' num2str(counter)]);
        texture2 = GetTextureFeatures(inCol(:,:,1)/100);
        
        TotalSuperpixels = max(label2(:));
        TotalColorSuperPixels = TotalColorSuperPixels + TotalSuperpixels;
        spStd2 = zeros(TotalSuperpixels, 4);
        spCol2 = zeros(TotalSuperpixels, 4);
        spSIFT2 = zeros(TotalSuperpixels, fcount);
        spTexture2 = zeros(TotalSuperpixels, size(texture2, 3));

        grayValue = inCol(:,:,1)/100;
        colStd = stdfilt(grayValue, ones(stdSize));
        nnList2 = GetListofNeighbouringSuperPixels(label2, nnSize);
        ImgRow = size(inCol,1);     ImgCol = size(inCol,2);
        Sift2 = reshape(Sift2, [ImgRow*ImgCol fcount]);
        texture2 = reshape(texture2, [ImgRow*ImgCol size(texture2,3)]); 
        
        for I = 1:TotalSuperpixels
            clc;disp(['Processing Color Image ' num2str(counter) ': ' num2str(I) ' of ' num2str(TotalSuperpixels)]);
            sp = (label2 == I);
            % Compute the mean SURF Value of each Super Pixel.
%             for J = 1:fcount
%                 p = Sift2(:,:,J);
%                 spSIFT2(I, J) = mean(p(sp));        
%             end
            currSP_Idx = sp==1;
            currSP_sift = Sift2(currSP_Idx,:);
            spSIFT2(I,:) = mean(currSP_sift, 1);
            % Compute the mean Texture Value of each Super Pixel.
%             for J = 1:size(texture2, 3)
%                 p = texture2(:,:,J);
%                 spTexture2(I, J) = mean(p(sp));        
%             end   
            currSP_texture = texture2(currSP_Idx,:);
            spTexture2(I,:) = mean(currSP_texture, 1);
            % Compute the mean intensity value of each Super Pixel.
            spStd2(I, 1) = mean(grayValue(sp));
            spStd2(I, 2) = 0;
            % Compute the mean Standard Deviation value of each Super Pixel.
            spStd2(I, 3) = mean(colStd(sp));
            spStd2(I, 4) = 0;
            % Compute the mean Intensity and SD value of NN Super Pixels.
            for J = 1:nnList2{I, 2}
                spNN = nnList2{I, 3};
                spStd2(I, 2) = spStd2(I, 2) + mean(grayValue((label2 == spNN(J))));
                spStd2(I, 4) = spStd2(I, 4) + mean(colStd((label2 == spNN(J))));
            end 
            % Get the Super Pixel's centroid color value.
            spStd2(I, 2) = spStd2(I, 2) / nnList2{I, 2};
            spStd2(I, 4) = spStd2(I, 4) / nnList2{I, 2};
            [row col] = find(sp==1);
            minR = min(row);      maxR = max(row);
            minC = min(col);      maxC = max(col);
            rc2 = floor((maxR + minR)/2);
            cc2 = floor((maxC + minC)/2);     
            spCol2(I, 1) = rc2;
            spCol2(I, 2) = cc2;
            spCol2(I, 3) = inCol(rc2, cc2, 2);
            spCol2(I, 4) = inCol(rc2, cc2, 3);
        end        
        colImgFeatureCell{counter, 1} = spSIFT2;
        colImgFeatureCell{counter, 2} = spTexture2;
        colImgFeatureCell{counter, 3} = spStd2;
        colImgFeatureCell{counter, 4} = spCol2;
    end
    
	spStd2 = zeros(TotalColorSuperPixels, 4);
	spCol2 = zeros(TotalColorSuperPixels, 4);
	spSIFT2 = zeros(TotalColorSuperPixels, fcount);
	spTexture2 = zeros(TotalColorSuperPixels, 40);
    spCounter = 0;
    for counter = 1:size(inpColImg, 1)    
        spCount = size(colImgFeatureCell{counter, 1}, 1);
        spSIFT2(spCounter+1:spCounter+spCount, :) =  colImgFeatureCell{counter, 1};
        spTexture2(spCounter+1:spCounter+spCount, :) =  colImgFeatureCell{counter, 2};
        spStd2(spCounter+1:spCounter+spCount, :) =  colImgFeatureCell{counter, 3};
        spCol2(spCounter+1:spCounter+spCount, :) =  colImgFeatureCell{counter, 4};        
        spCounter = spCounter + spCount;
    end
    if ~exist(outputDir,'dir')
        dos(['mkdir ' outputDir]);
    end
    save(dataFileName, 'spTexture1', 'spTexture2', 'spStd1', 'spStd2', 'spSIFT1', 'spSIFT2', 'inImg', 'inCol', 'label1', 'label2', 'nnList1', 'nnList2', 'spCol2');
    %SelectedNN
end
% clear all;
load(dataFileName)
disp('computing NN...');
TotalSuperpixels = size(spTexture1, 1);
nnCount = min(TotalSuperpixels, SelectedNN);
% [D, N] = knn(spSIFT1', spSIFT2', nnCount);
% N = FilterBasedOnTexture(N, spTexture1, spTexture2, nnCount/2);
[costMatrix N] = knn(spTexture1', spTexture2', nnCount);
[costMatrix N] = FilterBasedOnSIFT(N, spSIFT1, spSIFT2, nnCount/2, costMatrix);
[costMatrix N] = FilterBasedOnIntensity(N, spStd1(:, 1:2), spStd2(:, 1:2), floor(nnCount/4), costMatrix, 0.2);
[costMatrix N] = FilterBasedOnIntensity(N, spStd1(:, 3:4), spStd2(:, 3:4), floor(nnCount/8), costMatrix, 0.1);

nnCount = floor(nnCount/8);
save([keyWord '_lab.mat'], 'inImg', 'inCol', 'label1', 'label2', 'N', 'costMatrix', 'nnCount', 'nnList1', 'spCol2');
test_median_Voting_New(keyWord);
disp('done....');
return;

function [costMatrix nnList] = FilterBasedOnSIFT(nnSIFT, spSIFT1, spSIFT2, numOfNN, costMatrix)
    TotalSuperpixels = max(size(nnSIFT));
    Y = zeros(size(nnSIFT, 1), size(spSIFT1, 2));
    
    nnList = zeros(numOfNN, TotalSuperpixels);
    for I=1:TotalSuperpixels
        X = spSIFT1(I, :);
        nnCost = zeros(size(costMatrix, 1), 1);
        
        for J=1:size(nnSIFT, 1)
            Y(J, :) = spSIFT2(nnSIFT(J, I), :);
        end
        [D, N] = knn(X', Y', numOfNN);
        for J=1:numOfNN
            nnList(J, I) = nnSIFT(N(J), I);
            nnCost(J) = costMatrix(N(J), I) * 0.2 + D(J) * 0.5;
        end
        costMatrix(:, I) = nnCost;
    end
return;

function [costMatrix nnList] = FilterBasedOnIntensity(nnSIFT, spInt1, spInt2, numOfNN, costMatrix, W)
    TotalSuperpixels = max(size(nnSIFT));
    Y = zeros(size(nnSIFT, 1), size(spInt1, 2));
    
    nnList = zeros(numOfNN, TotalSuperpixels);
    for I=1:TotalSuperpixels
        X = spInt1(I, :);
        nnCost = zeros(size(costMatrix, 1), 1);
        
        for J=1:size(nnSIFT, 1)
            Y(J, :) = spInt2(nnSIFT(J, I), :);
        end
        [D, N] = knn(X', Y', numOfNN);
        for J=1:numOfNN
            nnList(J, I) = nnSIFT(N(J), I);
            nnCost(J) = costMatrix(N(J), I) + D(J) * W;
        end
        costMatrix(:, I) = nnCost;
    end
return;


function Result = GetListofNeighbouringSuperPixels(labels, diskSize)

TotalSuperpixels = max(labels(:));
Result = cell(TotalSuperpixels, 3);

se = strel('disk', diskSize);
for I = 1:TotalSuperpixels
    sp = (labels == I);
    spd = imdilate(sp, se);
    spNN = unique(labels(spd));
    spNN = spNN(spNN ~= I);
    Result{I, 1} = I;
    Result{I, 2} = size(spNN, 1);
    Result{I, 3} = spNN;
end

return;

function [ScribbledImg, labels, colorMatrix] = GenerateColorInfo(inImg, inSmImg, wrappedColImg, mask, spCount)
wrappedGrayImg = rgb2gray(wrappedColImg);
% Compute the superpixels by using QuickSeg.
% [phi, boundary, seg, labels] = superpixels(inImg, spCount);
 [labels,~] = superpixels(inImg, spCount);

labels = GetSingleComponent(labels);
labels(~inSmImg) = -1; 
% seg(~inSmImg) = 1;
% figure; imshow(seg);
% imwrite(seg, 'tmp\superpixel.png');
[ScribbledImg colorMatrix] = GetScribbleImg(labels, inImg, wrappedGrayImg, wrappedColImg, mask);
% figure; imshow(ScribbledImg);
% imwrite(ScribbledImg, 'tmp\ScribbledImg.png');
return;

% Function will ensure that each lable consists of a single component.
function labels = GetSingleComponent(inputLabels)
labels = inputLabels;
labels(:) = -1;

UniqueLabelIdx = unique(inputLabels);
UniqueLabelIdx(UniqueLabelIdx==-1) = [];

numLabel = length(UniqueLabelIdx);
currNewIdx = 0;

for ii = 1:numLabel
    currLabel = UniqueLabelIdx(ii);
    tmpImg = inputLabels == currLabel;
    [L, num] = bwlabel(tmpImg);
    if num == 0
        error('Error. Quick sift has no component.');
    end
    for jj = 1:num
        currNewIdx = currNewIdx + 1;
        labels(L==jj) = currNewIdx;
    end
end
return;

function [r g b] = GetColorofSuperPixel(currLabelBinaryMap, InputGrayImg, ...
                                            WrappedGrayImg, wrappedColorImg)
r = 0; g = 0; b = 0;
% Mask the superpixel to Get the Intensity Information from Internet Image.
WrappedGrayImg(~currLabelBinaryMap) = 0;
InputGrayImg(~currLabelBinaryMap) = 0;
ImageDiff = abs(InputGrayImg - WrappedGrayImg);
minDiff = min(ImageDiff(currLabelBinaryMap));

[row col] = find(currLabelBinaryMap==1);
minR = min(row);      maxR = max(row);
minC = min(col);      maxC = max(col);
x = 0;  y = 0;
for i = minR:maxR
    for j = minC:maxC
        if currLabelBinaryMap(i, j)
            if ImageDiff(i, j) == minDiff
                x = i;  y = j;
            end
        end
    end
end
if 1%(x > 0) & (y > 0)
    r = wrappedColorImg(x, y, 1);
    g = wrappedColorImg(x, y, 2);
    b = wrappedColorImg(x, y, 3);
end    
return;                                                   

function [ScribbledColorsImg colorMatrix] = GetScribbleImg(InputLabel, InputGrayImg, ...
                                            wrappedGrayImg, wrappedColorImg, mask)
% Compute the number of Labels.
uniqueLabel = unique(InputLabel);
uniqueLabel(uniqueLabel==-1) = [];
% Number of super pixels in the gray image.
% Each label represents one super pixel.
numSuperPixels = length(uniqueLabel); 
colorMatrix = zeros(numSuperPixels, 3);

ScribbledSeedsColorsImg_r = InputGrayImg;   
ScribbledSeedsColorsImg_g = InputGrayImg;   
ScribbledSeedsColorsImg_b = InputGrayImg;   

SEScribble = strel('disk', 7);

for currSuperPixelLabel = 1:numSuperPixels
    % Get the binary mask of the current super pixel.
    currLabelBinaryMap = InputLabel == uniqueLabel(currSuperPixelLabel);
    % Need to call one function here that will get the color of that 
    % super pixel from wrapped image/s.
    if sum(currLabelBinaryMap(:)) == 0
        continue;
    end
    
    isWrapp = mask & currLabelBinaryMap;
    if sum(isWrapp(:)) == 0
        tmpMap = currLabelBinaryMap;
        while sum(isWrapp(:)) == 0
            tmpMap = imdilate(tmpMap, SEScribble);
            isWrapp = mask .* tmpMap;
        end
        tmpMap = mask & tmpMap;
        [rgb_r rgb_g rgb_b] = GetColorofSuperPixel(tmpMap, ...
                                                    InputGrayImg, ...
                                                    wrappedGrayImg, ...
                                                    wrappedColorImg);
    else
        [rgb_r rgb_g rgb_b] = GetColorofSuperPixel(currLabelBinaryMap, ...
                                                        InputGrayImg, ...
                                                        wrappedGrayImg, ...
                                                        wrappedColorImg);
    end
    %   Getting the center of the super pixel, while ensuring that the
    %   center of the super pixel is within the super pixel.
    [r c] = find(currLabelBinaryMap==1);
    [currCenter MaxDiff] = GetMedoid([r c]);
    ScribbledSeedsColorsImg_r(currCenter(1), currCenter(2)) = rgb_r;
    ScribbledSeedsColorsImg_g(currCenter(1), currCenter(2)) = rgb_g;
    ScribbledSeedsColorsImg_b(currCenter(1), currCenter(2)) = rgb_b;

    %   Assigning more pixels within the super pixel colors.
    currLabelBinaryMap = imerode(currLabelBinaryMap,SEScribble);  
    colorMatrix(currSuperPixelLabel, 1) = rgb_r;
    colorMatrix(currSuperPixelLabel, 2) = rgb_g;
    colorMatrix(currSuperPixelLabel, 3) = rgb_b;
    ScribbledSeedsColorsImg_r(currLabelBinaryMap) = rgb_r;
    ScribbledSeedsColorsImg_g(currLabelBinaryMap) = rgb_g;
    ScribbledSeedsColorsImg_b(currLabelBinaryMap) = rgb_b;
end
ScribbledColorsImg = cat(3, ScribbledSeedsColorsImg_r, ...
                                 ScribbledSeedsColorsImg_g, ...
                                 ScribbledSeedsColorsImg_b);
return;

%   Function will find the medoid from Data.
%   Data is a N x D matrix, where N is the number of feature vectors and D
%   is the dimension of each feature vector.
function [currSuperPixelColor MaxColorSqDiff] = GetMedoid(Data)
numData = size(Data,1);
%   Finding the distance between itself and its neighbors.
DistMatrix = zeros(numData,numData);
for currIdx = 1:numData
    currFV = Data(currIdx,:);
    nextFV = Data;
    currFV = repmat(currFV, numData, 1);
    
    currDist = abs(currFV - nextFV);
    currDist = sum(currDist,2);
    DistMatrix(:,currIdx) = currDist;
end
%   Finding the total distance between itself and all its neighbor.
DistLinearMatrix = sum(DistMatrix);
[minDist minIdx] = min(DistLinearMatrix);
currSuperPixelColor = Data(minIdx,:);
MaxColorSqDiff = max(DistMatrix(minIdx,:));
return;

function ScribbledColorsImg = CreateScribbleImg(inImg, labels, bpColor, clustNum, colorMatrix)


% Number of super pixels in the gray image.
% Each label represents one super pixel.
numSuperPixels = size(clustNum, 1); 

ScribbledSeedsColorsImg_r = inImg;   
ScribbledSeedsColorsImg_g = inImg;   
ScribbledSeedsColorsImg_b = inImg;   

SEScribble = strel('disk', 7);

for currSuperPixelLabel = 1:numSuperPixels
    % Get the binary mask of the current super pixel.
    currLabelBinaryMap = labels == currSuperPixelLabel;
    % Need to call one function here that will get the color of that 
    % super pixel from wrapped image/s.
%     if sum(currLabelBinaryMap(:)) == 0
%         continue;
%     end

    %   Getting the center of the super pixel, while ensuring that the
    %   center of the super pixel is within the super pixel.
    rgb_r = colorMatrix(bpColor(currSuperPixelLabel), 1);
    rgb_g = colorMatrix(bpColor(currSuperPixelLabel), 2);
    rgb_b = colorMatrix(bpColor(currSuperPixelLabel), 3);
    
    [r c] = find(currLabelBinaryMap==1);
    [currCenter MaxDiff] = GetMedoid([r c]);
    ScribbledSeedsColorsImg_r(currCenter(1), currCenter(2)) = rgb_r;
    ScribbledSeedsColorsImg_g(currCenter(1), currCenter(2)) = rgb_g;
    ScribbledSeedsColorsImg_b(currCenter(1), currCenter(2)) = rgb_b;

    %   Assigning more pixels within the super pixel colors.
    currLabelBinaryMap = imerode(currLabelBinaryMap,SEScribble);  
    colorMatrix(currSuperPixelLabel, 1) = rgb_r;
    colorMatrix(currSuperPixelLabel, 2) = rgb_g;
    colorMatrix(currSuperPixelLabel, 3) = rgb_b;
    ScribbledSeedsColorsImg_r(currLabelBinaryMap) = rgb_r;
    ScribbledSeedsColorsImg_g(currLabelBinaryMap) = rgb_g;
    ScribbledSeedsColorsImg_b(currLabelBinaryMap) = rgb_b;
end
ScribbledColorsImg = cat(3, ScribbledSeedsColorsImg_r, ...
                                 ScribbledSeedsColorsImg_g, ...
                                 ScribbledSeedsColorsImg_b);
return;


function InputTextureFeatures= GetTextureFeatures(InputGrayImg)

%   Step 0. Smoothing before getting responses.
InputGrayImg = imfilter(InputGrayImg,fspecial('gaussian',5,1.),'same','replicate');
%   Step 1. Doing for the input image.
InputTextureFeatures = OwnGetGaborResponse(InputGrayImg);
%   Getting the Texture image based on texton.
InputTextureImg = GetInputTextureImg(InputTextureFeatures);
return;

function FastTextonImg = GetInputTextureImg(TextureFeatures)

if size(TextureFeatures,3) == 40
    load('Texton.mat');
else
    load('SIFT.mat');
    Texton = SIFT;
end

ImgRow = size(TextureFeatures,1);
ImgCol = size(TextureFeatures,2);
NumDim = size(TextureFeatures,3);

%   Error check.
if NumDim ~= size(Texton,2) %#ok<NODEF>
    error('Error. Number of dimension of texton is different from Texton.mat file.');
end

NumTexton = size(Texton,1);

%   Reshaping for easy computation.
%   ImgTextureResponse is a M x D matrix, where M is the number of image
%   pixels, and D is the number of dimensions.
NumImgPixels = ImgRow*ImgCol;
ImgTextureResponse = reshape(TextureFeatures, NumImgPixels, NumDim);

%   Set ImgTextureResponse to be [D x N].
ImgTextureResponse = ImgTextureResponse';
%   Set ImgTextureResponse to be [D x N].
Texton = Texton';
%   Fast nearest neighbor.
[FastDistMatrix FastMinIdx] = knn(ImgTextureResponse, Texton, 1);
TmpFastMinIdx = FastMinIdx(1,:);
FastTextonImg = reshape(TmpFastMinIdx, ImgRow, ImgCol);
return;
