function qualityscore = DFE_score(imdist)
%% feature extraction
[feat, time] = dfe_feature(imdist);

%% Quality Score Computation
load('model_LIVE');
qualityscore = predict(model,feat);

end