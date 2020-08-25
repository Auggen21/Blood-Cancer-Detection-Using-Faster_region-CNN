
img = imread('inp_red\ALL_5.bmp.png');
load('detector.mat');

[bbox, score, label] = detect(detector,img)

detectedImg = insertShape(img,'Rectangle',bbox);
figure
imshow(detectedImg)



%check various images, accuracy is low as it has only 30 images per class