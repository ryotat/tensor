% exp_postproc - performs CP decomposition after overlapped approach
%
% See also
%  tensorconst_adm
% 
% Reference
% "Estimation of low-rank tensors via convex optimization"
% Ryota Tomioka, Kohei Hayashi, and Hisashi Kashima
% arXiv:1010.0789
% http://arxiv.org/abs/1010.0789
%
% "Statistical Performance of Convex Tensor Decomposition"
% Ryota Tomioka, Taiji Suzuki, Kohei Hayashi, Hisashi Kashima
% NIPS 2011
% http://books.nips.cc/papers/files/nips24/NIPS2011_0596.pdf
%
% Convex Tensor Decomposition via Structured Schatten Norm Regularization
% Ryota Tomioka, Taiji Suzuki
% NIPS 2013
% http://papers.nips.cc/paper/4985-convex-tensor-decomposition-via-structured-schatten-norm-regularization.pdf
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt


addpath ver3.1/

load amino.mat

X0=permute(reshape(X,DimX), [2,3,1]);


sz=size(X0);
nn=prod(sz);

ntr=round(.5*nn);

tol=1e-3;
dparafac=4;

% Options for tucker/parafac
Options(4)=2;     % no scaling
Options(5)=100;   % display every 100 iterations


%
% Method 1: first train "const" and apply parafac
%
ind=randperm(nn); ind=ind(1:ntr)';
ind_test=setdiff(1:prod(sz), ind);
[I,J,K]=ind2sub(sz,ind);
yy=X0(ind);
t0=cputime;
[X,Z,Y,fval,gvals]=tensorconst_adm(zeros(sz),{I,J,K},yy,0,'tol', tol);

% Estimate the rank
r1 = numcomp(svd(Z{1}));
r2 = numcomp(svd(Z{2}));
r3 = numcomp(svd(Z{3}));

% Extract factors
[U1,S,V]=pca(Z{1},r1,20); ss=diag(S);
[U2,S,V]=pca(Z{2},r2,20);
[U3,S,V]=pca(Z{3},r3,20);
C=kolda3(X,U1',U2',U3');
F=parafac(C,dparafac, Options);
time(3)=cputime-t0;
err(3)=norm(X(ind_test)-X0(ind_test))/norm(X0(:));
U1=flipsign(U1*F{1});
U2=flipsign(U2*F{2});
U3=flipsign(U3*F{3});
figure
subplot(3,3,3);
plot(EmAx, U1, 'linewidth',2); grid on; 
title('Proposed(4)','fontsize',16);
subplot(3,3,6);
plot(ExAx, U2, 'linewidth',2); grid on; 
subplot(3,3,9);
plot(1:5, U3, 'linewidth',2); grid on; 
 
%
% Method 2: directly apply parafac
%
yfact=std(yy)*10;
Xobs=zeros(sz);
Xobs(ind)=X0(ind)/yfact;
Xobs(ind_test)=nan;
t0=cputime;
Factors=parafac(Xobs,dparafac,Options);
time(2)=cputime-t0;
Xp=nmodel(Factors)*yfact;
err(2)=norm(Xp(ind_test)-X0(ind_test))/norm(X0(:));
subplot(3,3,2);
plot(EmAx, Factors{1}*yfact, 'linewidth',2);
title('PARAFAC(4)','fontsize',16);
grid on; 
subplot(3,3,5);
plot(ExAx, Factors{2}, 'linewidth',2);
grid on; 
subplot(3,3,8);
plot(1:5, Factors{3}, 'linewidth',2);
grid on; 


%
% Method 0: True parafac
%
t0=cputime;
F0=parafac(Xobs, 3, Options);
time(1)=cputime-t0;
Xp=nmodel(Factors)*yfact;
err(1)=norm(Xp(ind_test)-X0(ind_test))/norm(X0(:));
subplot(3,3,1);
plot(EmAx, F0{1}*yfact, 'linewidth',2);
grid on; 
title('PARAFAC(3)','fontsize',16);
ylabel('Emission loadings','fontsize',16);
subplot(3,3,4);
plot(ExAx, F0{2}, 'linewidth',2);
grid on; 
ylabel('Excitation loadings','fontsize',16);
subplot(3,3,7);
plot(1:5, F0{3}, 'linewidth',2);
grid on; 
ylabel('Sample loadings','fontsize',16);


set(gcf,'papersize',[20 20]);

