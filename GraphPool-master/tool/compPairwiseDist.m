
function [ CorVMat] = compPairwiseDist( feats , nViews )
%GETIMDBPAIRSUFFLED Summary of this function goes here
% this function generates pair wise shuffled images structure 
% from original images struct and the CNN's relu7 features

%   Detailed explanation goes here
%   Each object has nViews and they are each passed into 
%   GetCorrelations for nViews * 4096 feat matrix to Get correlation matrix
%   add up each obj's corvMat to acquire global corvMat
%   output the corvmat for further graph clustering
%   Chu Wang
CorVMat = [];

nShapes = size(feats,1) / nViews;

CorMatUni = zeros(nViews,nViews);

for i = 1:nShapes
    if mod(i, 1000) == 0
        disp(['parsing ' num2str(i) '-th shape...'])
    end
    
    % get feat for current shape
    feat_tmp = feats((i-1)*nViews+1:i*nViews, :);
    % get correlation matrix and paired sequence
    [ CorMat_tmp] = GetCorrelations( feat_tmp , 'innerprod' );
    %[ CorMat_tmp] = GetCorrelations( feat_tmp , 'cosine' );
    
 
%    if  (std(reshape(CorMat_tmp,1,[]))  > 300)
%         std(reshape(CorMat_tmp,1,[]))  
%         CorMatUni = CorMatUni + CorMat_tmp;
%    end

%  tmp = sort(unique(CorMat_tmp -  diag(diag(CorMat_tmp))));
%    if  (std(tmp(2:end))  < 120)
%         std(tmp(2:end)) 
%         CorMatUni = CorMatUni + CorMat_tmp;
%    end
   
   
   CorMatUni = CorMatUni + CorMat_tmp;
    
    
end
% global minimizing
CorMatUni = CorMatUni /nShapes;
CorVMat = CorMatUni;


end


