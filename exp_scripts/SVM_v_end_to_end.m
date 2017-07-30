feats_file =  'feats/DS_relu6_endtoend.mat';

load(feats_file);
nViews = 12;




% svm 
test_idx  = images.set(:)== 3;
train_idx= ~test_idx;
test_idx = test_idx(1:nViews:end);
train_idx = train_idx(1:nViews:end);
feats_train_ready = feats(train_idx,:);
feats_test_ready = feats(test_idx,:);


labels = double (images.class(1:nViews:end));
trainLabel = labels(train_idx)';
testLabel = labels(test_idx)';
trainFeat = sparse(double(feats_train_ready));
testFeat = sparse ( double(feats_test_ready));

% train/test
cmd = [''];
cmd = [cmd ' -q'];
model = liblinear_train(trainLabel,trainFeat,cmd);
cmd = [''];
disp(['Training set accuracy:']);
[~,accuTrain,~] = liblinear_predict(trainLabel,trainFeat,model,cmd);
disp(['Test set accuracy:']);
[predTest,accuTest,decTest] = liblinear_predict(testLabel,testFeat,model,cmd);

