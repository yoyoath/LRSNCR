function x=WNNM_CL21(b,lambda,theta,method)

% "Low-Rank and Sparse Matrix Decomposition With Non-convex 
% Regularized for Hyperspectral Anomaly Detection"_ WEI YAO 

n=size(b,1);
x=zeros(n,1);
switch method
    %% g(x)=lambda*(x^theta)
    case 'Lp-norm'       
        for i=1:n
            if lambda*theta*b(i)^(theta-1)==0
                x(i)=b(i);
            else
                x0=b(i);
                while 1
                    x_new=b(i)-lambda*theta*(x0^(theta-1));
                    if x_new<0
                        x(i)=0;
                        break;
                    end
                    if abs(x_new-x0)<1e-5
                        x(i)=x_new;
                        break;
                    else
                        x0=x_new;
                    end
                end
            end
            tmp1=0.5*(b(i)^2);
            tmp2=0.5*(b(i)-x(i))^2+lambda*(x(i)^theta);
            if(tmp1<tmp2)
                x(i)=0;
            end 
            
        end
   %% g(x)=lambda*x when x<=lambda; (-x^2+2*lambda*theta*x-lambda^2)/2*(theta-1) when lambda<x<=theta*lambda; 
    %      =lambda^2*(theta+1)/2 when x>theta*lambda            theta>2
    case 'SCAD'
        n=length(b);
        temp1= sign(b).*min(lambda,max(abs(b)-lambda,0));
        temp2= sign(b).*min(theta*lambda,max(lambda,(abs(b)*(theta-1)-theta*lambda)/(theta-2)));
        temp3= sign(b).*max(theta*lambda,abs(b));
        temp=[temp1,temp2,temp3]';
        obj1=0.5*(temp1-b).^2+lambda*abs(temp1);
        obj2=0.5*(temp2-b).^2+(-temp2.^2+2*theta*lambda*abs(temp2)-(lambda)^2)/(2*(theta-1));
        obj3=0.5*(temp3-b).^2+lambda^2*(theta+1)/2;
        obj_all=[obj1';obj2';obj3'];                
        [obj_min,ind2]=min(obj_all);
        x=temp(ind2+(0:3:3*(n-1)))';
        
%% g(x)=lambda*log(theta*x+1)/log(theta+1)   theta is very small postive value 
    case 'LSP'
         for i=1:n
             dgb=lambda*theta/(theta*b(i)+1)/log(theta+1);
             if dgb==0
                 x(i)=b(i);
             else
                 x0=b(i);
                 while 1
                    x_new=b(i)-lambda*theta/(theta*b(i)+1)/log(theta+1);
                    if x_new<0
                        x(i)=0;
                        break;
                    end
                    if abs(x_new-x0)<1e-5
                        x(i)=x_new;
                        break;
                    else
                        x0=x_new;
                    end
                end
            end
            tmp1=0.5*(b(i)^2);
            tmp2=0.5*(b(i)-x(i))^2+lambda*log(theta*x(i)+1)/log(theta+1);
            if(tmp1<tmp2)
                x(i)=0;
            end                         
         end
     %% g(x)= 
    case 'MCP'
        for i=1:n
            if b(i)<lambda*theta
                dgb=lambda-(b(i)/theta);
            else
                dgb=0;
            end
            if dgb==0
             x(i)=b(i);
            else
             x0=b(i);
                while 1
                    if x0<theta*lambda
                        dgx=lambda-x0/theta;
                    else
                        dgx=0;                    
                    end
                    x_new=b(i)-dgx;
                    if x_new<0
                        x(i)=0;
                        break;
                    end
                    if abs(x_new-x0)<1e-5
                        x(i)=x_new;
                        break;
                    else
                        x0=x_new;
                    end
                end
            end
            tmp1=0.5*(b(i)^2);
            if x(i)<theta*lambda
                tmp2=0.5*(b(i)-x(i))^2+lambda*x(i)-x(i)^2/(2*lambda);
            else
                tmp2=0.5*(b(i)-x(i))^2+0.5*theta*lambda*lambda;
            end
            if(tmp1<tmp2)
                x(i)=0;
            end                         
        end
        
    case 'Geman'
       for i=1:n
           dgb=lambda*theta/((b(i)+theta)^2);
           if dgb==0
               x(i)=b(i);
           else
                x0=b(i);
                while 1
                    dgx=lambda*theta/((x0+theta)^2);
                    x_new=b(i)-dgx;
                    if x_new<0
                        x(i)=0;
                        break;
                    end
                    if abs(x_new-x0)<1e-5
                        x(i)=x_new;
                        break;
                    else
                        x0=x_new;
                    end
                end
           end
           tmp1=0.5*(b(i)^2);
           tmp2=0.5*(b(i)-x(i))^2+lambda*x(i)/(x(i)+lambda);
           if(tmp1<tmp2)
                x(i)=0;
            end     
       end
    case 'Laplace'
       for i=1:n
           dgb=(lambda/theta)*exp(-b(i)/theta);
           if dgb==0
               x(i)=b(i);
           else
                x0=b(i);
                while 1
                    dgx=(lambda/theta)*exp(-x0/theta);
                    x_new=b(i)-dgx;
                    if x_new<0
                        x(i)=0;
                        break;
                    end
                    if abs(x_new-x0)<1e-5
                        x(i)=x_new;
                        break;
                    else
                        x0=x_new;
                    end
                end
           end
           tmp1=0.5*(b(i)^2);
           tmp2=0.5*(b(i)-x(i))^2+lambda*(1-exp(-x(i)/theta));
           if(tmp1<tmp2)
                x(i)=0;
            end     
       end
    case 'capped_L1'      
        x1=max(theta,b);
        x2=min(theta,max(0,b-lambda));
        x_all=[x1';x2'];
        h1=0.5*(x1-b).^2+lambda*theta;
        h2=0.5*(x2-b).^2+lambda*x2;
        [obj_min,ind]=min([h1';h2']);
        x=x_all(ind+(0:2:2*(n-1)))';
    case 'WNNM' 
        %C=lambda
        temp=(b-eps).^2-4*(lambda-eps*b);
        ind=find (temp>0);
        x(ind)=max(b(ind)-eps+sqrt(temp(ind)),0)/2;
    case 'WSNM' 
        %C=lambda p=theta
%         Temp         =   sqrt(max( b.^2 - n*lambda^2, 0 ));
%         for i=1:4
%             W_Vec    =   (lambda*sqrt(n))./( Temp.^(1/theta) + eps );               % Weight vector           
%             s1       =   solve_Lp_w(b, W_Vec, theta);
%             Temp     =   s1;
%         end
        
        W_vec=lambda*theta*b.^(theta-1);%(lambda*n)./(b+eps);
        for iter=1:4        
            for i=1:n
                dgb=W_vec(i)*theta*b(i)^(theta-1);
                if dgb==0
                   x(i)=b(i);
                else
                    x0=b(i);
                    while 1
                        dgx=W_vec(i)*theta*x0^(theta-1);
                        x_new=b(i)-dgx;
                        if x_new<0
                            x(i)=0;
                            break;
                        end
                        if abs(x_new-x0)<1e-5
                            x(i)=x_new;
                            break;
                        else
                            x0=x_new;
                        end
                    end
                end
                tmp1=0.5*(b(i)^2);
                tmp2=0.5*(b(i)-x(i))^2+lambda*(W_vec(i)*x(i)^theta);
                if(tmp1<tmp2)
                    x(i)=0;
                end     
            end
            W_vec=lambda*theta*x.^(theta-1);%(lambda*n)./(x+eps);
        end
           
        
    case 'L1'
        x=max(b-lambda,0);
    otherwise
        disp('method error!')
end
end