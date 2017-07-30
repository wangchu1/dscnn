function [ ReluFeat ] = full_unit(  ReluFeat , nShapes , pool_type  )
%FULL_UNIT Summary of this function goes here
%   Detailed explanation goes here
%   full stride pooling 
%   Chu Wang


stride = size(ReluFeat,1) / nShapes;

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

