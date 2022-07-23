function [feat, runtime] = dfe_feature(imdist)
%------------------------------------------------
% Feature Computation
% imdist should be double format
% scalenum: number of scales
%-------------------------------------------------
tic;
PCA_pth = 'PCA/';
imdist = double(imdist);
[hgt_org, wdt_org, ~] = size(imdist);
blk_sz_set = [0 2 4];
%% O
O{1} = 0.3*imdist(:,:,1) + 0.04*imdist(:,:,2) - 0.35*imdist(:,:,3);
O{2} = 0.34*imdist(:,:,1) - 0.6*imdist(:,:,2) + 0.17*imdist(:,:,3);
O{3} = 0.06*imdist(:,:,1) + 0.63*imdist(:,:,2) + 0.27*imdist(:,:,3);
%% KLT
blk_sz = 2;
load([PCA_pth,'Kernels/PCA_kernel_x',num2str(blk_sz),'_bmp']);
hgt = floor(hgt_org/blk_sz)*blk_sz;
wdt = floor(wdt_org/blk_sz)*blk_sz;
for idx_c = 1:3
    featO{idx_c} = [];
    X = im2col(O{idx_c}(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
    coef = X*kernel{idx_c};
    for i = 2:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,i));
        featO{idx_c} = [featO{idx_c}, alpha, overallstd];
    end   
end
feat_KLT = [featO{1} featO{2} featO{3}];
%% PC
for idx_c = 1:3
    featO{idx_c} = [];
    O_PC = phasecong3(O{idx_c});
    scalenum = 2;
    for itr_scale = 1 : scalenum
        blk_sz = blk_sz_set(itr_scale);
        if blk_sz>1
            load([PCA_pth,'Kernels/PCA_kernel_PC_x',num2str(blk_sz),'_bmp']);
            hgt = floor(hgt_org/blk_sz)*blk_sz;
            wdt = floor(wdt_org/blk_sz)*blk_sz;
            X = im2col(O_PC(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
            coef = X*kernel{idx_c};
        else
            X = O_PC;
            coef = X(:);
        end
        tmp = coef(:,1);
        tmp = tmp(abs(tmp)>0.01);
        featO{idx_c} = [featO{idx_c}, wblfit(tmp)];
        for i = 2:size(coef,2)
            [alpha, overallstd] = estimateggdparam(coef(:,i));
            featO{idx_c} = [featO{idx_c}, alpha, overallstd];
        end
    end    
end
feat_PC = [featO{1} featO{2} featO{3}];
%% GM
dx = [1 -1];
dy = dx';
for idx_c = 1:3
    featO{idx_c} = [];
    Ix = conv2(O{idx_c}, dx, 'same');
    Iy = conv2(O{idx_c}, dy, 'same');
    O_G = sqrt(Ix.^2 + Iy.^2);
    scalenum = 2;
    for itr_scale = 1 : scalenum
        blk_sz = blk_sz_set(itr_scale);
        if blk_sz>1
            load([PCA_pth,'Kernels/PCA_kernel_G_x',num2str(blk_sz),'_bmp']);
            hgt = floor(hgt_org/blk_sz)*blk_sz;
            wdt = floor(wdt_org/blk_sz)*blk_sz;
            X = im2col(O_G(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
            coef = X*kernel{idx_c};
        else
            X = O_G;
            coef = X(:);
        end
        tmp = coef(:,1);
        tmp = tmp(abs(tmp)>0.01);
        featO{idx_c} = [featO{idx_c}, wblfit(tmp)];
        for i = 2:size(coef,2)
            [alpha, overallstd] = estimateggdparam(coef(:,i));
            featO{idx_c} = [featO{idx_c}, alpha, overallstd];
        end
    end    
end
feat_GM = [featO{1} featO{2} featO{3}];
%% MSCN
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
for idx_c = 1:3
    featO{idx_c} = [];
    mu = filter2(window, O{idx_c}, 'same');
    sigma = sqrt(abs(filter2(window, O{idx_c}.*O{idx_c}, 'same') - mu.*mu));
    O_MSCN = (O{idx_c}-mu)./(sigma+1);
    scalenum = 3;
    for itr_scale = 1 : scalenum
        blk_sz = blk_sz_set(itr_scale);
        if blk_sz>1
            load([PCA_pth,'Kernels/PCA_kernel_MSCN_x',num2str(blk_sz),'_bmp']);
            hgt = floor(hgt_org/blk_sz)*blk_sz;
            wdt = floor(wdt_org/blk_sz)*blk_sz;
            X = im2col(O_MSCN(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
            coef = X*kernel{idx_c};
        else
            X = O_MSCN;
            coef = X(:);
        end
        for i = 1:size(coef,2)
            [alpha, overallstd] = estimateggdparam(coef(:,i));
            featO{idx_c} = [featO{idx_c}, alpha, overallstd];
        end
    end    
end
feat_MSCN = [featO{1} featO{2} featO{3}];
%%
feat = [feat_KLT feat_PC feat_GM feat_MSCN].^0.5;
feat(isnan(feat)) = 0;
feat(isinf(feat)) = 10;
runtime = toc;
end