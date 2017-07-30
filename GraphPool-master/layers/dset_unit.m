function [ feats_res , C ] = dset_unit(  feats , nViews , criteria ,poolMethod , clusterAssigned )
%DSET_UNIT Summary of this function goes here
%   pooling unit associated with dominant set clustering
%   we first compute the dominant set from the acquired affinity matrix
%   between views.
%   We then pool features according to the cluster
%   Dominant set clustering and pooling unit
%   Chu Wang

nShapes = size(feats,1) / nViews;
if isempty(clusterAssigned)
    % compute affinity mat
    filename = [ 'DominantSetLibrary-master/' num2str(nViews) 'views_CorvMat_dot.mat' ];
    if exist(filename , 'file') == 2 && nViews > 24
        load(filename , 'CorMat');
    else
        [ CorMat] = compPairwiseDist( feats , nViews );
    end

    %[ CorMat] = compPairwiseDist( feats , nViews );
    
    if nViews > 2
        AffinityMat = ( CorMat - min(min(CorMat)) ) / (max(max(CorMat)) - min(min(CorMat)));
    else
        AffinityMat = CorMat;
    end
    
    %AffinityMat = CorMat/(max(max(CorMat)));
    %AffinityMat = CorMat;
    % for dissimilar clusters
    if strcmp(criteria , 'dissimilar')
        AffinityMat = 1 ./(AffinityMat + min(min(AffinityMat(AffinityMat(:) > 0))));
    elseif strcmp(criteria , 'similar')
        if nViews <=12
            for i = 1:nViews
                AffinityMat(i,i) = 0;
            end
        else
            for i = 1:nViews
                % promote fewer clusters
                AffinityMat(i,i) = -1;
            end
        end
        
    else
        disp('unknown criteria for clustering');
    end
    
    % dominant set clustering
    % disp('Computing cluster...')
    %[C]=dominantset(AffinityMat,[],[],1e-8,[],0 ,[], [],3);
    [C]=dominantset(AffinityMat,[],[],1e-8,[],0 ,[], [],nViews);
    
    % finetune on the dominant large cluster if any [deleted]
    
    disp(clusterAssigned);
else
    disp('using assigned cluster...')
    disp(clusterAssigned);
    C = clusterAssigned;
end


C_id = unique(C);

% acquire feats for each cluster
selector = [];
feats_pooled = [];
for idx = 1:length(C_id)
    cur_id = C_id(idx);
    selecton = C(:) == cur_id;
    selector = repmat(selecton , nShapes,1 );
    feats_cur = feats(selector , :);
    % call vpool unit 
    [ feats_cur_pooled ] = vpool_unit( feats_cur , sum(selecton) , poolMethod  );
    feats_pooled = [feats_pooled , feats_cur_pooled];
end

% rearrange feats_pooled from (nShapes , #C*dim ) to (nShapes*#C , dim )
% for consistency with later cascaded layers
% for a single graph/shape the process is bit different.
if nShapes == 1
    feats_res =  feats_pooled';
else
    feats_res =  reshape(feats_pooled , [nShapes ,size(feats_pooled,2) /length(C_id), length(C_id)   ]);
    feats_res = permute(feats_res,[3,1,2]);
    feats_res =  reshape(feats_res , [nShapes * length(C_id) , size(feats_pooled,2) /length(C_id)  ]);
end




end



% finetune dominant cluster 
%     [val,freq] =mode(C);
%     max_val = max(C);
%     
%     if freq / nViews > 0.5
%         disp('finetuning clusters');
%         % select these view nodes that belongs to the dominent cluster
%         sel = C(:) == val;
%         A_tmp = AffinityMat;
%         A_tmp(~sel, :) = [];
%         A_tmp(:,~sel) = [];
%         % rescale the affinity matrix to enhance the gap between clusters
%         A_tmp = exp(A_tmp+1);
%         % normalization
%         A_tmp = ( A_tmp - min(min(A_tmp)) ) / (max(max(A_tmp)) - min(min(A_tmp)));
%         for i = 1:freq
%             A_tmp(i,i) = 0;
%         end
%         
%         % redo clustering
%         [C_tmp]=dominantset(A_tmp,[],[],1e-8,[],0);
%         % assign new sub clusters back to original clustering.
%         C_tmp = C_tmp + max_val;
%         C(sel) = C_tmp;
%     end
