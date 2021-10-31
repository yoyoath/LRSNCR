function [IDX,C,inv_Sigma]=autokmeans(X)
[dim,num]=size(X);
IDX=zeros(num,6);

for k=2:7
IDX(:,k-1)=kmeans(X',k,'Start','plus');
% n_i=zeros(k,1);
%     for i=1:k
%         n_i(i)=sum(IDX==i);
%     end
% temp=C-repmat(m_',[k,1]);
% crit(k-1)=((num-k)/(k-1))*sum(sum(temp.^2,2).*n_i)/sum(SUMD);

end
eva = evalclusters(X',IDX,'CalinskiHarabasz'); 
K=eva.OptimalK+1;
[IDX,C,SUMD]=kmeans(X',K,'Start','plus');
inv_Sigma=zeros(dim,dim,K);
for i=1:K
    temp2=[];
    for j=1:num
        if IDX(j)==i
            temp2=[temp2,j];
        end
    end
    cur=X(:,temp2);
    inv_Sigma(:,:,i)=inv(cov(cur'));
end
end