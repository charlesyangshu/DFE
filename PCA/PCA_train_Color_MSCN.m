clear;clc;close all;

blk_sz = 4;
im_pth = '../../../Dataset/pristinedata/pristine/';
current = pwd;
cd(sprintf('%s',im_pth))
names        = ls;
names        = names(3:end,:);
cd(current);
%%
X_Y  = [];
X_Y2 = [];
X_Y3 = [];
%% MSCN
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
%%
for idx_im = 1:size(names,1)
    tic
    im_name = ['i',num2str(idx_im)]

    img_org = imread([im_pth,names(idx_im,:)]);
    img_org = double(img_org);
    
    %% O
    O1 = 0.3*img_org(:,:,1) + 0.04*img_org(:,:,2) - 0.35*img_org(:,:,3);
    O2 = 0.34*img_org(:,:,1) - 0.6*img_org(:,:,2) + 0.17*img_org(:,:,3);
    O3 = 0.06*img_org(:,:,1) + 0.63*img_org(:,:,2) + 0.27*img_org(:,:,3);
    %% RGB
    O1 = img_org(:,:,1);
    O2 = img_org(:,:,2);
    O3 = img_org(:,:,3);
    %% MSCN
    mu = filter2(window, O1, 'same');
    sigma = sqrt(abs(filter2(window, O1.*O1, 'same') - mu.*mu));
    O1 = (O1-mu)./(sigma+1);
    
    mu = filter2(window, O2, 'same');
    sigma = sqrt(abs(filter2(window, O2.*O2, 'same') - mu.*mu));
    O2 = (O2-mu)./(sigma+1);
    
    mu = filter2(window, O3, 'same');
    sigma = sqrt(abs(filter2(window, O3.*O3, 'same') - mu.*mu));
    O3 = (O3-mu)./(sigma+1);
    %%
    [hgt, wdt, d] = size(img_org);
    hgt = floor(hgt/blk_sz)*blk_sz;
    wdt = floor(wdt/blk_sz)*blk_sz;

    blk1 = im2col(O1(1:hgt,1:wdt),[blk_sz blk_sz],'distinct')';
    blk1 = blk1 - mean(blk1,2);%Mean_Remove
    blk2 = im2col(O2(1:hgt,1:wdt),[blk_sz blk_sz],'distinct')';
    blk2 = blk2 - mean(blk2,2);%Mean_Remove
    blk3 = im2col(O3(1:hgt,1:wdt),[blk_sz blk_sz],'distinct')';
    blk3 = blk3 - mean(blk3,2);%Mean_Remove
    %%
    struct_idx1 = (std(blk1')>0)';
    blk1 = blk1(struct_idx1,:);
    struct_idx2 = (std(blk2')>0)';
    blk2 = blk2(struct_idx2,:);
    struct_idx3 = (std(blk3')>0)';
    blk3 = blk3(struct_idx3,:);
    %%
    X_Y  = [X_Y; blk1];
    X_Y2 = [X_Y2; blk2];
    X_Y3 = [X_Y3; blk3];
    toc
end
%% PCA Training
[V,S] = pca(X_Y);
vt = V(:,1:end-1);
v0 = ones(blk_sz*blk_sz,1)/blk_sz;
kernel{1} = [v0, vt];
%%
[V,S] = pca(X_Y2);
vt = V(:,1:end-1);
v0 = ones(blk_sz*blk_sz,1)/blk_sz;
kernel{2} = [v0, vt];
%%
[V,S] = pca(X_Y3);
vt = V(:,1:end-1);
v0 = ones(blk_sz*blk_sz,1)/blk_sz;
kernel{3} = [v0, vt];
%%
save(['Kernels/PCA_kernel_MSCN_x',num2str(blk_sz),'_bmp'],'-v7.3','kernel');





