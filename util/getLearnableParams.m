function params = getLearnableParams(net)
%GETLEARNABLEPARAMS Gets all the learnable params of a net as a list.
    params = [];
    for i = 1:numel(net.Learnables)
        params = [params; net.Learnables.Value{i}(:)];
    end
end

