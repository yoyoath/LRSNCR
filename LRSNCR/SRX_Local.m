function D=SRX_Local(Data,r_in,r_out)
[H,W,dim]=size(Data);
X=reshape(Data, H*W, dim)';
[IDX,~,inv_Sigma]=autokmeans(X);

IDX=reshape(IDX,[H,W]);
D=zeros(H*W,1);
tmp2=zeros(H,W);
for i=1:H
    i
    if i<=r_in
            top_in=1;
            bottom_in=2*r_in+1;
    elseif i>H-r_in
        top_in=H-2*r_in;
        bottom_in=H;
    else
        top_in=i-r_in;
        bottom_in=i+r_in;
    end
    if i<=r_out
            top_out=1;
            bottom_out=2*r_out+1;
    elseif i>H-r_out
        top_out=H-2*r_out;
        bottom_out=H;
    else
        top_out=i-r_out;
        bottom_out=i+r_out;
    end
    for j=1:W
        if j<=r_in
            left_in=1;
            right_in=2*r_in+1;
        elseif j>W-r_in
            left_in=W-2*r_in;
            right_in=W;
        else
            left_in=j-r_in;
            right_in=j+r_in;
        end
        if j<=r_out
            left_out=1;
            right_out=2*r_out+1;
        elseif j>W-r_out
            left_out=W-2*r_out;
            right_out=W;
        else
            left_out=j-r_out;
            right_out=j+r_out;
        end
        temp=Data(top_out:top_in-1,left_out:right_out,:);
        temp1=reshape(temp,[],dim);
        temp=Data(bottom_in+1:bottom_out,left_out:right_out,:);     
        temp2=reshape(temp,[],dim);
        temp=Data(top_in:bottom_in,left_out:left_in-1,:);     
        temp3=reshape(temp,[],dim);
        temp=Data(top_in:bottom_in,right_in+1:right_out,:);     
        temp4=reshape(temp,[],dim);
        X_in=[temp1;temp2;temp3;temp4];
        m_X_in=mean(X_in);
        tmp=squeeze(Data(i,j,:))-m_X_in';
        tmp2(i,j)=tmp'*inv_Sigma(:,:,IDX(i,j))*tmp;
        
        

    end
end
% for i=1:H
%     if i==1
%         top=1;
%         bottom=3;
%     elseif i==H
%         top=H-2;
%         bottom=H;
%     else
%         top=i-1;
%         bottom=i+1;
%     end
%     for j=1:W
%          if j==1
%             left=1;
%             right=3;
%         elseif j==W
%             left=W-2;
%             right=W;
%         else
%             left=j-1;
%             right=j+1;
%          end
%         X_in=[];
%         for s=top:bottom
%             for t=left:right
%                 if s==i&&t==j
%                     continue;
%                 end
%                 X_in=[X_in,Data(s,t,:)];
%             end
%         end
%         m_X_in=mean(squeeze(X_in));
%         tmp=squeeze(Data(i,j,:))-m_X_in';
%         tmp2(i,j)=tmp'*inv_Sigma(:,:,IDX(i,j))*tmp;
%     end
% end
D=reshape(tmp2,[1,H*W]);
end