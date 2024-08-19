function net = updateDropoutRate(net, idx, p)
%UPDATEDROPOUTRATE Summary of this function goes here
    layer = net.Layers(idx);
    newLayer = dropoutLayer(p, 'Name', layer.Name);
    net = replaceLayer(net, layer.Name, newLayer);
end

