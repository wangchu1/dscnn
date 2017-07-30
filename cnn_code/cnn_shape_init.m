function net = cnn_shape_init(classNames, varargin)
% original code adopted from MVCNN git hub.
% customized by Chu Wang to support cluster-pooling recurrence

opts.base = 'imagenet-matconvnet-vgg-m'; 
opts.restart = false; 
opts.nViews = 12; 
opts.viewpoolPos = 'relu5'; 
opts.viewpoolType = 'max'; 
%opts.weightInitMethod = 'xavierimproved';
opts.weightInitMethod = 'gaussian';
opts.scale = 1;
opts.cluster_pool = false;
opts.cluster = {};
opts.clusterpoolType = {};
opts.poolsquare = 1;
opts.networkType = 'simplenn'; % only simplenn is supported currently
opts = vl_argparse(opts, varargin); 


assert(strcmp(opts.networkType,'simplenn'), 'Only simplenn is supported currently'); 

init_bias = 0.1;
nClass = length(classNames);

% Load model, try to download it if not readily available
if ~ischar(opts.base), 
  net = opts.base; 
else
  netFilePath = fullfile('data','models', [opts.base '.mat']);
  if ~exist(netFilePath,'file'),
    fprintf('Downloading model (%s) ...', opts.base) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models/', ...
      [opts.base '.mat']), netFilePath) ;
    fprintf(' done!\n');
  end
  net = load(netFilePath);
end
assert(strcmp(net.layers{end}.type, 'softmax'), 'Wrong network format'); 
dataTyp = class(net.layers{end-1}.weights{1}); 

% Initiate the last but one layer w/ random weights
widthPrev = size(net.layers{end-1}.weights{1}, 3);
nClass0 = size(net.layers{end-1}.weights{1},4);
if nClass0 ~= nClass || opts.restart, 
  net.layers{end-1}.weights{1} = init_weight(opts, 1, 1, widthPrev, nClass, dataTyp);
  net.layers{end-1}.weights{2} = zeros(nClass, 1, dataTyp); 
end

% Initiate other layers w/ random weights if training from scratch is desired
if opts.restart, 
  w_layers = find(cellfun(@(c) isfield(c,'weights'),net.layers));
  for i=w_layers(1:end-1), 
    sz = size(net.layers{i}.weights{1}); 
    net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
    net.layers{i}.weights{2} = zeros(sz(4), 1, dataTyp); 
  end	
end

% Swap softmax w/ softmaxloss
net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;
    

% recurrent cluster-pooling structure; 
% 3-step construction because typical cluster convergence pattern.
if opts.cluster_pool == true && length(opts.clusterpoolType) == 3, 
  ClusterPoolLayer1 = struct('name', 'cluster_pool1', ...
    'type', 'custom', ...
    'nViews',length(opts.cluster{1}),...
    'cluster', opts.cluster{1}, ...
    'method', opts.clusterpoolType{1}, ...
    'forward', @cluster_pool_fw, ...
    'backward', @cluster_pool_bw);

  ClusterPoolLayer2 = struct('name', 'cluster_pool2', ...
    'type', 'custom', ...
    'nViews',length(opts.cluster{2}),...
    'cluster', opts.cluster{2}, ...
    'method', opts.clusterpoolType{2}, ...
    'forward', @cluster_pool_fw, ...
    'backward', @cluster_pool_bw);

  viewpoolLayer = struct('name', 'viewpool', ...
    'type', 'custom', ...
    'vstride', length(opts.cluster{3}), ...
    'method', opts.clusterpoolType{3}, ...
    'forward', @viewpool_fw, ...
    'backward', @viewpool_bw);
 % read from bottom to top
 % cluster pool 1 -> cluster pool 2-> vpool
 

  net = modify_net(net, viewpoolLayer, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);    
  net = modify_net(net, ClusterPoolLayer2, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);
  net = modify_net(net, ClusterPoolLayer1, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);


end

% model ablation study 
if opts.cluster_pool == true && length(opts.clusterpoolType) == 2, 
  ClusterPoolLayer1 = struct('name', 'cluster_pool1', ...
    'type', 'custom', ...
    'nViews',length(opts.cluster{1}),...
    'cluster', opts.cluster{1}, ...
    'method', opts.clusterpoolType{1}, ...
    'forward', @cluster_pool_fw, ...
    'backward', @cluster_pool_bw);

  viewpoolLayer = struct('name', 'viewpool', ...
    'type', 'custom', ...
    'vstride', length(opts.cluster{2}), ...
    'method', opts.clusterpoolType{2}, ...
    'forward', @viewpool_fw, ...
    'backward', @viewpool_bw);
 % read from bottom to top
 % cluster pool 1 -> cluster pool 2-> vpool
  net = modify_net(net, viewpoolLayer, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);    
  net = modify_net(net, ClusterPoolLayer1, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);

end



% update meta data
net.meta.classes.name = classNames;
net.meta.classes.description = classNames;

% speial case: when no class names specified, remove fc8/prob layers
if nClass==0, 
    net.layers = net.layers(1:end-2);
end
    
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end


% create DS cluster layer.
% -------------------------------------------------------------------------
function res_ip1 = cluster_pool_fw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
% 1,1,4096,b*v
[sz1, sz2, sz3, sz4] = size(res_i.x);

% grouping nodes in one cluster
C_id = unique(layer.cluster);
batch_size = sz4/layer.nViews;

% acquire feats for each cluster
selector = [];
feats_pooled = [];
for idx = 1:length(C_id)
    cur_id = C_id(idx);
    selecton = layer.cluster(:) == cur_id;
    selector = repmat(selecton , batch_size , 1 );
    feats_cur = res_i.x(:,:,:, selector);
    % pool
    vstride = sum(selecton);
    if strcmp(layer.method, 'avg'),
        feats_cur = permute(...
            mean(reshape(feats_cur,[sz1 sz2 sz3 vstride batch_size ]), 4), ...
            [1,2,3,5,4]);
    elseif strcmp(layer.method, 'max'),
        feats_cur = permute(...
            max(reshape(feats_cur,[sz1 sz2 sz3 vstride batch_size]), [], 4), ...
            [1,2,3,5,4]);
    else
        error('Unknown viewpool method: %s', layer.method);
    end
    % 1,1,#C*4096, B
    if idx == 1
        feats_pooled = feats_cur;
    else
        feats_pooled = cat(3, feats_pooled, feats_cur);
    end
    
end

% shape to 1,1,4096, B*#C
% consistent with input/output
res_ip1.x = reshape(reshape(permute(feats_pooled,[3 , 4, 1, 2]) , [sz3 , length(C_id), batch_size ,sz1,sz2 ]) , [sz1, sz2 , sz3 , length(C_id) * batch_size]);

end


% -------------------------------------------------------------------------
function res_i = cluster_pool_bw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
% 1 , 1 , 4096 , B*#Cluster
[sz1, sz2, sz3, sz4] = size(res_ip1.dzdx);
% 1 , 1 , 4096 , B*InputNodes
[sz1_fw, sz2_fw, sz3_fw, sz4_fw] = size(res_i.x);

input_dzdx = zeros(sz1_fw, sz2_fw, sz3_fw, sz4_fw, 'single');
input_dzdx = gpuArray(input_dzdx);

% grouping nodes in one cluster
C_id = unique(layer.cluster);
batch_size = sz4_fw/layer.nViews;

% acquire feats for each cluster
% acquire feats for each cluster
selector = [];
feats_pooled = [];
num_cluster = length(C_id);
% traverse each cluster
for idx = 1:length(C_id)
    cur_id = C_id(idx);
    selecton = layer.cluster(:) == cur_id;
    selector = repmat(selecton , batch_size , 1 );
    
    % feats from this cluster, before pooling
    feats_cur = res_i.x(:,:,:, selector);
    % dzdx wrt this cluster
    ResCluster_dzdx = res_ip1.dzdx(:,:,:, idx:num_cluster:sz4);
    % #ele in cluster
    vstride = sum(selecton);

    % reverse pooling effect wrt gradients
    if strcmp(layer.method, 'avg'),
        
        FromCluster_dzdx = ...
            reshape(repmat(reshape(ResCluster_dzdx / vstride, ...
            [sz1 sz2 sz3 1 batch_size]), ...
            [1 1 1 vstride 1]),...
            [sz1 sz2 sz3 vstride*batch_size]);
        
    
    elseif strcmp(layer.method, 'max'),
        % feats_cur 1 1 4096 B*#eleInCluster
        % max pool result: 1 1 4096 B
        % find which ele in cluster gives actual max response
        [~,I] = max(reshape(permute(feats_cur,[4 1 2 3]), ...
            [vstride, batch_size*sz1*sz2*sz3]),[],1);
        % finding maxed indices
        Ind = zeros(vstride,batch_size*sz1*sz2*sz3, 'single');
        Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
        Ind = permute(reshape(Ind,[vstride*batch_size,sz1,sz2,sz3]),[2 3 4 1]);
        
        % map dzdx wrt this cluster back to cluster space
        % backed dim: 1 1 4096 B*#eleIncluster
        % consistent with feats_cur
        FromCluster_dzdx = ...
            reshape(repmat(reshape(ResCluster_dzdx, ...
            [sz1 sz2 sz3 1 batch_size]), ...
            [1 1 1 vstride 1]),...
            [sz1 sz2 sz3 vstride*batch_size]) .* Ind;
        
        
        
        
    else
        error('Unknown viewpool method: %s', layer.method);
    end
    % FromCluster_dzdx -> original
    input_dzdx(:,:,:, selector) = FromCluster_dzdx;

    
    
end
res_i.dzdx = input_dzdx;
clear input_dzdx;
end



% -------------------------------------------------------------------------
function res_ip1 = viewpool_fw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_i.x);
if mod(sz4,layer.vstride)~=0, 
    error('all shapes should have same number of views');
end
if strcmp(layer.method, 'avg'), 
    res_ip1.x = permute(...
        mean(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'max'), 
    res_ip1.x = permute(...
        max(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), [], 4), ...
        [1,2,3,5,4]);
    % size(res_ip1.x)
elseif strcmp(layer.method, 'cat'),
    res_ip1.x = reshape(res_i.x,[sz1 sz2 sz3*layer.vstride sz4/layer.vstride]);
elseif strcmp(layer.method, 'eigen'),
    max_EVec = single(1);
    res_ip1.x = reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]);
    % create singleton dimension for numofBatch
    res_ip1.x = permute(res_ip1.x , [1,2,3,5, 4]);
    % 4096* nViews* nBatch   
    B = res_ip1.x;
    res_ip1.x = single([]);
    for i = 1:size(B,4)
        BatchViewData = squeeze(B(:,:,:,i,:));
        SigmaMatrix = BatchViewData' * BatchViewData;
        % eigen system analysis
        [EVectors,DValue] = eig(single(SigmaMatrix));
        % this creation makes the dimension (x,y,4096,nViews,nBatches)
        res_ip1.x(1,1,:,:,i) =  single(single(BatchViewData) * single(EVectors(:,end+1-max_EVec:end)) );
        
    end
    % permute it to make dimension (x,y,4096,nBatches,nViews)
    res_ip1.x = single(permute( res_ip1.x , [1,2,3,5,4]));
else
    error('Unknown viewpool method: %s', layer.method);
end

end


% -------------------------------------------------------------------------
function res_i = viewpool_bw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_ip1.dzdx);
if strcmp(layer.method, 'avg'), 
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx / layer.vstride, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]);
% back propagation for max pooling.   
elseif strcmp(layer.method, 'max'), 
    [~,I] = max(reshape(permute(res_i.x,[4 1 2 3]), ...
                [layer.vstride, sz4*sz1*sz2*sz3]),[],1);
    % finding maxed indices
    Ind = zeros(layer.vstride,sz4*sz1*sz2*sz3, 'single');
    Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
    Ind = permute(reshape(Ind,[layer.vstride*sz4,sz1,sz2,sz3]),[2 3 4 1]);
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]) .* Ind;
elseif strcmp(layer.method, 'cat'),
    res_i.dzdx = reshape(res_ip1.dzdx, [sz1 sz2 sz3/layer.vstride sz4*layer.vstride]);
elseif strcmp(layer.method, 'eigen'),
    max_EVec = 1;
    res_i.dzdx = [];
    B = permute( reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4]), [1,2,3,5, 4] );
    % create singleton dimension for numofBatch

    for i = 1:sz4
        BatchViewData = squeeze(B(:,:,:,i,:));
        
        SigmaMatrix = BatchViewData' * BatchViewData;
        % eigen system analysis
        [EVectors,DValue] = eig(SigmaMatrix);
        
        % multiply dzdx by EV transpose
        res_i.dzdx(1,1,:,:,i) = single( squeeze(res_ip1.dzdx(:,:,:,i)) * EVectors(:,end+1-max_EVec:end)') ;
        % 4096 * maxEvec
        % res_ip1.x(1,1,:,:,i) =  BatchViewData * EVectors(:,end+1-max_Evec:end);
    end
    size(res_i.dzdx)
    res_i.dzdx = single(reshape(res_i.dzdx ,  [sz1 sz2 sz3 layer.vstride*sz4])) ;
    size(res_i.dzdx)
else
    error('Unknown viewpool method: %s', layer.method);
end

end
