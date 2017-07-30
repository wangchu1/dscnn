classdef Conv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    initMethod = 'gaussian'
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      switch obj.initMethod
        case 'gaussian'
          params{1} = randn(obj.size,'single') * sc ;
        case 'one'
          params{1} = permute(eye(obj.size(3),'single'),[3 4 1 2]);
      end
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Conv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
      switch obj.initMethod
        case 'gaussian'
        case 'one'
          assert(obj.size(1)==1 && obj.size(2)==1, ['Filter width should be set to ''1'' for ' ...
            'assigned initialization method (''one'')']);
          assert(obj.size(3)==obj.size(4), ['Initialization method (''one'') '...
            'requires the 3rd and 4th dimensions of the filter weights to be equal!']);
        otherwise
          error('Unsupported initialization method: ', obj.initMethod);
      end
    end

  end
end
