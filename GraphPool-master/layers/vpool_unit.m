function [ ReluFeat ] = vpool_unit( ReluFeat , stride , pool_type  )
%VPOOL_AVG Summary of this function goes here
%   Detailed explanation goes here
%   View pooling layer using average pooling/ max pooling.
%   Input size: ReluFeat (nViews * nShape, reluDim)
%   Stride: pooling stride with regard to views
%   pool_method : 'max' or 'avg'
%   Chu Wang


if strcmp(pool_type , 'max')
    
    ReluFeat = reshape(ReluFeat, [stride, size(ReluFeat,1)/stride , size(ReluFeat,2)]);
    ReluFeat = max(ReluFeat,[],1);
    ReluFeat = squeeze(ReluFeat);
    
elseif strcmp(pool_type , 'avg')
    
    ReluFeat = reshape(ReluFeat, [stride, size(ReluFeat,1)/stride , size(ReluFeat,2)]);
    ReluFeat = mean(ReluFeat,1);
    ReluFeat = squeeze(ReluFeat);
    
else 
    error('unknown pooling unit specification.');
end

end



