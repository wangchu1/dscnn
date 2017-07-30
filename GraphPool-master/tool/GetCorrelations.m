function [ CorMatrix ] = GetCorrelations( feat , method )
%GETCORRELATIONS Summary of this function goes here
%   Detailed explanation goes here

% feat matrix with size nViews by 4096
% Chu Wang


sz = size(feat,1);
CorMatrix = [];

for i = 1:sz
    for j = 1:sz
        if strcmp(method, 'innerprod')
            CorMatrix(i,j) = dot(feat(i,:), feat(j,:));
        elseif strcmp(method , 'cosine')
            CorMatrix(i,j) = dot(feat(i,:), feat(j,:)) / (norm(feat(i,:)) * norm(feat(j,:)));      
        else
            error('unknown similarity metric');
        end
        
    end
end

end

