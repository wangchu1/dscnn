classdef ConvTranspose < dagnn.Layer
  properties
    size = [0 0 0 0]
    hasBias = true
    upsample = [1 1]
    crop = [0 0 0 0]
    initMethod = 'gaussian'
    numGroups = 1
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconvt(...
        inputs{1}, params{1}, params{2}, ...
        'upsample', obj.upsample, ...
        'crop', obj.crop, ...
        'numGroups', obj.numGroups, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconvt(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'upsample', obj.upsample, ...
        'crop', obj.crop, ...
        'numGroups', obj.numGroups, ...
        obj.opts{:}) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [...
        obj.upsample(1) * (inputSizes{1}(1) - 1) + obj.size(1) - obj.crop(1) - obj.crop(2), ...
        obj.upsample(2) * (inputSizes{1}(2) - 1) + obj.size(2) - obj.crop(3) - obj.crop(4), ...
        obj.size(4), ...
        inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      rfs.size = (obj.size(1:2) - 1) ./ obj.upsample + 1 ;
      rfs.stride = 1 ./ [obj.upsample] ;
      rfs.offset = (2*obj.crop([1 3]) - obj.size(1:2) + 1) ...
        ./ (2*obj.upsample) + 1 ;
    end

    function params = initParams(obj)
      % todo: test this initialization method
      sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      switch obj.initMethod
        case 'gaussian'
          params{1} = randn(obj.size,'single') * sc ;
        case 'bilinear'
          factorY = single([1:obj.upsample(1) (obj.upsample(1)-1):-1:1]/obj.upsample(1));
          factorX = single([1:obj.upsample(2) (obj.upsample(2)-1):-1:1]/obj.upsample(2));
          params{1} = bsxfun(@times,factorY'*factorX,permute(eye(obj.size(3)),[3 4 1 2]));
      end
      if obj.hasBias
        params{2} = zeros(obj.size(3),1,'single') * sc ;
      end
    end
    
    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function set.crop(obj, crop)
      if numel(crop) == 1
        obj.crop = [crop crop crop crop] ;
      elseif numel(crop) == 2
        obj.crop = crop([1 1 2 2]) ;
      else
        obj.crop = crop ;
      end
    end

    function set.upsample(obj, upsample)
      if numel(upsample) == 1
        obj.upsample = [upsample upsample] ;
      else
        obj.upsample = upsample ;
      end
    end

    function obj = ConvTranspose(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.upsample = obj.upsample ;
      obj.crop = obj.crop ;
      switch obj.initMethod
        case 'gaussian'
        case 'bilinear'
          assert(all(obj.size(1:2)==obj.upsample*2-1), 'Incompatible filter size!');
          assert(obj.size(3)==obj.size(4), ['Initialization method (''bilinear'') '... 
            'requires the 3rd and 4th dimensions of the filter weights to be equal!']);
        otherwise
          error('Unsupported initialization method: ',obj.initMethod);
      end
      if(sum(obj.crop(1:2))+obj.upsample(1)~=obj.size(1) || ...
         sum(obj.crop(3:4))+obj.upsample(2)~=obj.size(2)), 
        warning(['Output won''t be exactly [%d %d] times of input. Consider set ' ...
          '''crop'' to [%d %d %d %d] instead.'], ...
          obj.upsample(1), obj.upsample(2), ...
          ceil((obj.size(1)-obj.upsample(1))/2), floor((obj.size(1)-obj.upsample(1))/2), ...
          ceil((obj.size(2)-obj.upsample(2))/2), floor((obj.size(2)-obj.upsample(2))/2));
      end
    end
  end
end
