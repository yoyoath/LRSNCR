function [A_hat,E_hat ,iter] = inexact_alm_Nonconvex_HAD(D,lowrank_method,C,theta,sparse_method,p1,p2)

% D - m x n matrix of observations/data (required input)
%
% Support - observation data indicator (binary matrix) 
%
% C - parameter for the weight setting in the weighted nuclear norm
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(L,S,Y,u) = |L|_w,* + |S|_Capped_2,1 + <Y,A-L-S> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (A-L-S);
%   \mu = \rho * \mu;
% end
% addpath PROPACK;

[m n] = size(D);
tol=1e-7;
maxIter = 1000;


% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) ;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros( m, n);
E_hat = zeros( m, n);
mu = 1/norm_two; % this one can be tuned
mu_bar = mu * 1e7;
rho = 1.05;          % this one can be tuned
d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;
lambda = 1 / sqrt(m);
while ~converged       
    iter = iter + 1;  
    temp_T = D - A_hat + (1/mu)*Y;
    norm_temp_T=sqrt(sum(temp_T.^2,2));
    shrink_norm=LRSNCR(norm_temp_T,p1/mu,p2,sparse_method);
    for i=1:m
        if shrink_norm(i)==0
            E_hat(i,:)=zeros(1,n);
        else
            E_hat(i,:)=temp_T(i,:)*shrink_norm(i)/norm_temp_T(i);
        end
    end
%     E_hat = max(temp_T - lambda/mu, 0);
%     E_hat = E_hat+min(temp_T + lambda/mu, 0);

    if choosvd(n, sv) == 1
        [U S V] = lansvd(D - E_hat + (1/mu)*Y, sv, 'L');
    else
        [U S V] = svd(D - E_hat + (1/mu)*Y, 'econ');
    end

   diagS = diag(S);
   tempDiagS=LRSNCR(diagS,C/mu,theta,lowrank_method);
   svp=length(tempDiagS>0);
   A_hat = U(:,1:svp)*diag(tempDiagS(1:svp))*V(:,1:svp)';  

    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end


    total_svd = total_svd + 1;
  
    Z = D - A_hat - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
%         disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
