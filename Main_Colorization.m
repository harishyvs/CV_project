function Main_Colorization
close all;
warning off;

% Number of Super Pixels ::
param.spCount = 2000;

keyWord = 'cv16';

param.inRatio = 1;
param.colRatio = 1;
param.nnSize = 660;
param.RecomputationRequired = 1;
inpColImg = cell(1, 1);
InputImg = ['./Input/' keyWord '.jpg'];
inpColImg{1} = './Input/cv15.png';
Colorization(keyWord, InputImg, inpColImg, param);

return;