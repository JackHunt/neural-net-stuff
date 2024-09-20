function net = setLearnableParams(net, params)
%SETLEARNABLEPARAMS Sets all the learnable params of a net from a list.
    start_idx = 1;
    for i = 1:numel(net.Learnables)
        param_size = size(net.Learnables.Value{i});
        numel_param = numel(net.Learnables.Value{i});
        end_idx = start_idx + numel_param - 1;
        
        net.Learnables.Value{i} = reshape(params(start_idx:end_idx), param_size);
        
        start_idx = end_idx + 1;
    end
end

