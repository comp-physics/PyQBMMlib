function moment = Moment_Weights(abscissas,weights,ids)

total_points = size(weights,2);
moment = 0.0;
for ii=1:total_points
    moment = moment +weights(ii)*abscissas(1,ii)^(ids(1))*abscissas(2,ii)^(ids(2));
end


end