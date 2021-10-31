clear all;
close all;
clc;

%% Dataset
% ABU
load abu-urban-2;
mask=map;

% load TIR_data;
% load TIR_gth;
% data=ave_3;  

f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end
figure,imshow(f_show);imwrite(f_show,'im.jpg');
figure,imshow(mask,[]);imwrite(mask,'gt.jpg');
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

NBOXPLOT=zeros(H*W,6);

%%%%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
Y1=reshape(DataTest, num, Dim);
Y=reshape(DataTest, num, Dim)';
%%%%

%% Global RX
disp('GRX')
tic;
r1 = RX(Y);  % input: num_dim x num_sam    rx
toc
r_max = max(r1(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r1 > tau);
  PF1(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD1(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r1,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','RX'), imshow(f_show);imwrite(f_show,'RX.jpg');
area_RX = sum((PF1(1:end-1)-PF1(2:end)).*(PD1(2:end)+PD1(1:end-1))/2);
NBOXPLOT(:,1)=f_show(:);

%% Perform_SRX_Local
disp('LRX')
tic;
r2=SRX_Local(DataTest,9,27);
toc
r_max2 = max(r2(:));
taus = linspace(0, r_max2, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r2> tau);
  PF2(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD2(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r2,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','SRX_Local'), imshow(f_show);imwrite(f_show,'SRX_Local.jpg');
area_SRX_Local = sum((PF2(1:end-1)-PF2(2:end)).*(PD2(2:end)+PD2(1:end-1))/2);
NBOXPLOT(:,2)=f_show(:);

%% LRASR
disp('LRASR')
beta=0.1;
lamda=1;
Dict=ConstructionD_lilu(Y,15,20);
tic;
[S,E]=LRASR(Y,Dict,beta,lamda,1);
toc
r3=sqrt(sum(E.^2,1));
r_max3 = max(r3(:));
taus = linspace(0, r_max3, 5000);
PF3=zeros(1,5000);
PD3=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r3> tau);
  PF3(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD3(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_LRASR = sum((PF3(1:end-1)-PF3(2:end)).*(PD3(2:end)+PD3(1:end-1))/2);
f_show=reshape(r3,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LRASR'), imshow(f_show);imwrite(f_show,'LRASR.jpg');
NBOXPLOT(:,3)=f_show(:)';

%% LSMAD
disp('LSMAD')
tic;
[L,S,RMSE,error]=GoDec(Y',28,floor(0.0022*Dim)*9,2);
toc
L=L';
S=S';

mu=mean(L,2);
r4=(diag((Y-repmat(mu,[1,num]))'*pinv(cov(L'))*(Y-repmat(mu,[1,num]))))';

r_max4 = max(r4(:));
taus = linspace(0, r_max4, 5000);
PF4=zeros(1,5000);
PD4=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r4> tau);
  PF4(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD4(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_LSMAD = sum((PF4(1:end-1)-PF4(2:end)).*(PD4(2:end)+PD4(1:end-1))/2);
f_show=reshape(r4,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LSMAD'), imshow(f_show);imwrite(f_show,'LSMAD.jpg');
NBOXPLOT(:,4)=f_show(:);

%% RPCA-RX
disp('RPCA-RX')
tic;
[r0 ,Output_S, Output_L] = Unsupervised_RPCA_Detect_v1(DataTest);
toc
XS = reshape(Output_S, num, Dim);
r5 = RX(XS');  % input: num_dim x num_sam

r_max5 = max(r5(:));                                              
taus = linspace(0, r_max5, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r5 > tau);
  PF5(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD5(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r5,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','RPCA-RX'), imshow(f_show);imwrite(f_show,'RPCA-RX.jpg');
area_RPCA = sum((PF5(1:end-1)-PF5(2:end)).*(PD5(2:end)+PD5(1:end-1))/2);
NBOXPLOT(:,5)=f_show(:);

%% test
disp('Proposed')
tic;

[LS, XS,iter ]=inexact_alm_Nonconvex_HAD(Y1, 'WNNM',1e3,0,'capped_L1',1,10);
toc
r6=(sqrt(sum(XS.^2,2)))';
r_max6 = max(r6(:));                                                
taus = linspace(0, r_max6, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r6 >= tau);
  PF6(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD6(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r6,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LRSNCR'), imshow(f_show);imwrite(f_show,'LRSNCR.jpg');
NBOXPLOT(:,6)=f_show(:);
area_Nonconvex = sum((PF6(1:end-1)-PF6(2:end)).*(PD6(2:end)+PD6(1:end-1))/2);

figure,
plot(PF1, PD1, 'g-', 'LineWidth', 2);hold on; %GRX
plot(PF2, PD2, 'r-', 'LineWidth', 2);  %LRX
plot(PF3, PD3, 'y-', 'LineWidth', 2); %LRSASR
plot(PF4, PD4, 'c-', 'LineWidth', 2); %%LSMAD
plot(PF5, PD5, 'k-', 'LineWidth', 2); %RPCA-RX
plot(PF6, PD6, 'm-', 'LineWidth', 2); %proposed
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('GRXD','LRXD','LRASR','LSMAD','RPCA-RX','Proposed method');
axis([0 0.1 0 1]);hold off;

figure, 
semilogx(PF1, PD1, 'g-', 'LineWidth', 2);hold on;
semilogx(PF2, PD2, 'r-', 'LineWidth', 2);  
semilogx(PF3, PD3, 'y-', 'LineWidth', 2);
semilogx(PF4, PD4, 'c-', 'LineWidth', 2);
semilogx(PF5, PD5, 'k-', 'LineWidth', 2);
semilogx(PF6, PD6, 'm-', 'LineWidth', 2);
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('GRXD','LRXD','LRASR','LSMAD','RPCA-RX','Proposed method');
axis([0 0.1 0 1]);hold off;
