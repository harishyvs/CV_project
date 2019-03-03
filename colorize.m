function im = Colorize(inImg, ScribbledImg)

%set solver=1 to use a multi-grid solver 
%and solver=2 to use an exact matlab "\" solver
solver=2; 
[r c ch] = size(inImg);
if ch == 1
    I = zeros(r, c, 3);
    I(:, :, 1) = inImg;
    I(:, :, 2) = inImg;
    I(:, :, 3) = inImg;
    inImg = I;
end
gI = inImg;
cI = ScribbledImg;
colorIm=(sum(abs(gI-cI),3)>0.01);
colorIm=double(colorIm);

sgI=rgb2ntsc(gI);
scI=rgb2ntsc(cI);
   
ntscIm = [];
ntscIm(:,:,1)=sgI(:,:,1);
ntscIm(:,:,2)=scI(:,:,2);
ntscIm(:,:,3)=scI(:,:,3);


% max_d=floor(log(min(size(ntscIm,1),size(ntscIm,2)))/log(2)-2);
% iu=floor(size(ntscIm,1)/(2^(max_d-1)))*(2^(max_d-1));
% ju=floor(size(ntscIm,2)/(2^(max_d-1)))*(2^(max_d-1));
% id=1; jd=1;
% colorIm=colorIm(id:iu,jd:ju,:);
% ntscIm=ntscIm(id:iu,jd:ju,:);

if (solver==1)
  nI=getVolColor(colorIm,ntscIm,[],[],[],[],5,1);
  nI=ntsc2rgb(nI);
else
  nI=getColorExact(colorIm,ntscIm);
end

im= nI;
%figure; imshow(nI)

%imwrite(nI,out_name)
   
return;  

%Reminder: mex cmd
%mex -O getVolColor.cpp fmg.cpp mg.cpp  tensor2d.cpp  tensor3d.cpp
