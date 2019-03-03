function MeargeSuperPixel

load matlab.mat;
DrawSuperPixel(inImg, labels)
stdDev = stdfilt(inImg);
% figure; imshow(seg);
figure; imshow(stdDev);
MixSuperpixels(inImg, labels, stdDev);
return;

function DrawSuperPixel(inImg, labels)
map = stdfilt(labels);
map(map>0.1)=1;
map(map<1) = inImg(map<1);
figure; imshow(map);
return;

function MixSuperpixels(inImg, labels, stdDev)
spCount = 1;
se = strel('disk', 3);
TotalSuperpixels = max(labels(:));
for I = 1:TotalSuperpixels
    sp = (labels == I);
    spd = imdilate(sp, se);
    spNN = unique(labels(spd));
    spNN = spNN(spNN ~= I);
    spMean = mean(stdDev(labels == I));
    labels(labels == I) = spCount;
    for J = 1:size(spNN, 1)
        spMean1 = mean(stdDev(labels == spNN(J)));
        if abs(spMean - spMean1) < 0.05
            labels(labels == spNN(J)) = spCount;
        end
    end
    spCount = spCount + 1;
end
DrawSuperPixel(inImg, labels);
return;
