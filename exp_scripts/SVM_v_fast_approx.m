% load data;
feats_file =  'feats/fastapprox_relu7.mat';
load(feats_file);
nViews = 12;
flag_train = images.set(:)~=3;
flag_test = ~flag_train;
feats_train = feats(flag_train,:);
feats_test = feats(flag_test,:);

% comment lines to switch between DS cluster-pooling and mvcnn
layers = {'d-similar-max' , 'd-similar-avg' , 'f-max'};
%layers = {'f-max'};

[ feats_train_ready , clusterAssigned ] = graph_pool_unit( feats_train , layers , nViews , []);
[ feats_test_ready , clusterAssigned ] = graph_pool_unit( feats_test , layers , nViews , clusterAssigned);

% configure and save cluster assignments.
% The 3rd recurrence has 2 clusters only, so it converges.
% assign the two nodes to same cluster
clusterAssigned{3} = zeros(2,1)
save('clusterAssigned.mat','clusterAssigned');

% svm process.
test_idx  = images.set(:)== 3;
train_idx= ~test_idx;
test_idx = test_idx(1:nViews:end);
train_idx = train_idx(1:nViews:end);


labels = double (images.class(1:nViews:end));
trainLabel = labels(train_idx)';
testLabel = labels(test_idx)';
trainFeat = sparse(double(feats_train_ready));
testFeat = sparse ( double(feats_test_ready));

% training/testing
cmd = [''];
cmd = [cmd ' -q'];
model = liblinear_train(trainLabel,trainFeat,cmd);
cmd = [''];
disp(['Training set accuracy:']);
[~,accuTrain,~] = liblinear_predict(trainLabel,trainFeat,model,cmd);
disp(['Test set accuracy:']);
[predTest,accuTest,decTest] = liblinear_predict(testLabel,testFeat,model,cmd);

