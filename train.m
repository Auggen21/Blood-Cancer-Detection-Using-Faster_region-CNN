% for training the network 


%load the labelled images and groundtruth
load('frcnnimg.mat');

%intially the labeling is done using lincy's lap. so the file source have
%to be changed

oldPathDataSource = "D:\Users\Lenovo PC\Downloads\data\input\";
newPathDataSource = "inp_red\"; % if you want to use data from your computer change this 
alterPaths = {[oldPathDataSource newPathDataSource]};
unresolvedPaths = changeFilePaths(gTruth,alterPaths) % after this file source will be changed.

% Load a pretrained ResNet-50.
net = resnet50; 

%resnet 50 is a model similar to vgg16 with some light changes. 
%In CNN, i have already given explaination to basic layers such as convolution,
%pooling layer dense layer, batch normlization and softmax layer. So vgg16
%is a basic network with 16 layers, as the number of convolutional layers
%are increased, it will cause the network to become unstable, i have
%explained about a gradient descent algorithm for weight change, and it
%will become unstable due to vanishing gradient problem. So in resnet 50 
%residual connections are given to avoid this.  In matlab resnet 50 accepts
%the input size of 240x240x3, so we have already reshaped the data.

lgraph = layerGraph(net); % used to get the layer details of resnet50

% plot the entire resnet50
figure;
plot(lgraph)

% Remove the last 3 layers of resnet50. 
layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };
lgraph = removeLayers(lgraph, layersToRemove); % it will remove the last three layers.

% plot the resnet50 with removed layers
figure;
plot(lgraph)


% Specify the number of classes the network should classify.
numClasses = 2; % ALL and MM
numClassesPlusBackground = numClasses + 1;

% Define new classification layers.
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC') %3 classes, so 3 neurons
    softmaxLayer('Name', 'rcnnSoftmax') %softmax layer for getting the probability
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new object classification layers.
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC'); % so the new 3 layers will get connected to the resnet50, avg_pool is the name of last layer in resnet50 layer removed network

%we have two outputs , one is clasification result and other is the output
%box to be  predicted, so here it will predict the square boxes of each
%class. and it is connected to the avg_pool 

% Define the number of outputs of the fully connected layer.
numOutputs = 4 * numClasses;

% Create the box regression layers.
boxRegressionLayers = [
    fullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')
    rcnnBoxRegressionLayer('Name','rcnnBoxDeltas')
    ];

% Add the layers to the network.
lgraph = addLayers(lgraph, boxRegressionLayers);

% Connect the regression layers to the layer named 'avg_pool'.
lgraph = connectLayers(lgraph,'avg_pool','rcnnBoxFC');


%now we are adding the box input,input size set is 14 by 14 we have already drawn the boundary of each
%tumor  affected nucleus

% Select a feature extraction layer.
featureExtractionLayer = 'activation_40_relu';

% Disconnect the layers attached to the selected feature extraction layer.
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch2a');
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch1');

% Add ROI max pooling layer.
outputSize = [14 14];
roiPool = roiMaxPooling2dLayer(outputSize,'Name','roiPool');
lgraph = addLayers(lgraph, roiPool);

% Connect feature extraction layer to ROI max pooling layer.
lgraph = connectLayers(lgraph, featureExtractionLayer,'roiPool/in');

% Connect the output of ROI max pool to the disconnected layers from above.
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch2a');
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch1');

%we have connected the box input inbetween the resnet50. We now define
%anchor boxes with the assuption that input box can be of these sizes.

% Define anchor boxes.
anchorBoxes = [
    16 16
    32 16
    16 32
    32 32
    ];

%faster rcnn uses a region proposal network.

% Create the region proposal layer.
proposalLayer = regionProposalLayer(anchorBoxes,'Name','regionProposal');

lgraph = addLayers(lgraph, proposalLayer);


% Number of anchor boxes.
numAnchors = size(anchorBoxes,1);

% Number of feature maps in coming out of the feature extraction layer. 
numFilters = 1024;

%define the layers for region proposal network
rpnLayers = [
    convolution2dLayer(3, numFilters,'padding',[1 1],'Name','rpnConv3x3')
    reluLayer('Name','rpnRelu')
    ];

lgraph = addLayers(lgraph, rpnLayers);

% Connect to RPN to feature extraction layer.
lgraph = connectLayers(lgraph, featureExtractionLayer, 'rpnConv3x3');


% Add RPN classification layers.
rpnClsLayers = [
    convolution2dLayer(1, numAnchors*2,'Name', 'rpnConv1x1ClsScores')
    rpnSoftmaxLayer('Name', 'rpnSoftmax')
    rpnClassificationLayer('Name','rpnClassification')
    ];
lgraph = addLayers(lgraph, rpnClsLayers);

% Connect the classification layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1ClsScores');


% Add RPN regression layers.
rpnRegLayers = [
    convolution2dLayer(1, numAnchors*4, 'Name', 'rpnConv1x1BoxDeltas')
    rcnnBoxRegressionLayer('Name', 'rpnBoxDeltas');
    ];

lgraph = addLayers(lgraph, rpnRegLayers);

% Connect the regression layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1BoxDeltas');

% Connect region proposal network.
lgraph = connectLayers(lgraph, 'rpnConv1x1ClsScores', 'regionProposal/scores');
lgraph = connectLayers(lgraph, 'rpnConv1x1BoxDeltas', 'regionProposal/boxDeltas');

% Connect region proposal layer to roi pooling.
lgraph = connectLayers(lgraph, 'regionProposal', 'roiPool/roi');

% Show the network after adding the RPN layers.
figure
plot(lgraph)
ylim([30 42])

%these rpn layers will generate the region proposals, which means that it
%will identify the refgions where the object can be found..
%create the image datastore and its boundary boxes

[imds,blds]=objectDetectorTrainingData(gTruth);

trainingData=combine(imds,blds);

%training with gradient descent algorithm. minibatch size 1 means it take
%just 1 input at a time, and then adjust the gradient. intial learn rate
%=0.001, max epochs -maximum times the iteration to be done=50, verbose
%frequency is for display puropose

 options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 20, ...
      'VerboseFrequency', 20, ...
      'ValidationData',trainingData.UnderlyingDatastores,...
      'CheckpointPath', tempdir);
  
  detector = trainFasterRCNNObjectDetector(trainingData, lgraph, options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
    
    
%after this save the model, you can save the detector by right click on the
%variable detector

save('detector5.mat','detector')