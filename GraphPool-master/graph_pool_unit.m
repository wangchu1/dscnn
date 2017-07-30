function [ feats_out , clusterAssigned ] = graph_pool_unit( feats  , layers , nNodes , assigned)
%GRAPH_POOL_UNIT Summary of this function goes here
%   Perform graph pooling on *a single* representer graph
%   only supports clustering for graph coarsening
%   Chu Wang

nShapes = size(feats,1) / nNodes;
%nNodes = size(feats,1);

clusterAssigned = cell(size(layers));


for ind = 1:length(layers)
    layer_catelog = strsplit(layers{ind} , '-');
    % 'g' matching unit
    if   strcmp( layer_catelog{1} , 'g' )
         [ feats , seq] = matching_unit( feats , size(feats,1) / nShapes , layer_catelog{2} , [] );
         seqAssigned{ind} = seq;
    % 'd' dominant set unit     
    elseif strcmp( layer_catelog{1} , 'd' )
        % use assigned cluster if any.
        if isempty(assigned)
            [ feats , C ] = dset_unit(  feats , size(feats,1) / nShapes , layer_catelog{2} , layer_catelog{3}  , [] );
        else
            [ feats , C ] = dset_unit(  feats , size(feats,1) / nShapes , layer_catelog{2} , layer_catelog{3}  , assigned{ind} );
        end
        
        clusterAssigned{ind} = C;
    % 'p' pooling unit    
    elseif strcmp( layer_catelog{1} , 'p' )
         [ feats ] = vpool_unit( feats , str2num(layer_catelog{3}) , layer_catelog{2}  );
    % 'f' full stride unit     
    elseif strcmp( layer_catelog{1} , 'f' )     
         [ feats ] = full_unit( feats , nShapes , layer_catelog{2}  );
         % specially handles a single graph scenario
         if nShapes == 1
            feats = feats';
         end
         
    end

end

% display message of cluster coarsening structure
message = ['cluster coarsening: ' num2str(nNodes ) ' -> '];
for ind = 1:length(layers)
    layer_catelog = strsplit(layers{ind} , '-');
    if strcmp( layer_catelog{1} , 'd' )
        message =  [ message num2str(length(unique(clusterAssigned{ind})))  ];
        message = [ message  '->'];
    end
    
end
message = [ message  '1'];

disp(message);

feats_out = feats;

end

