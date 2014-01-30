% exp_completion_l1 - performs experiment on sparse+low decomposition
%
% See also
%  exp_completion, tensorl1_adm, matrixl1_adm, plot_completion_l1
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


nrep=10;
sz=[50,50,50];
dims=[7,7,7];

trfrac=0.05:0.05:0.95;
noisefr = 0.1;  % 10% contaminated by noise
lambda=[exp(linspace(log(0.001),log(100),20))',...
        exp(linspace(log(1),log(40),20))'*ones(1,2)];



methods = {'constraint','matrix', 'tensor'};
% methods = {'constraint'}

for ll=1:nrep
  X0=randtensor3(sz,dims);
  nn=prod(size(X0));

  for kk=1:length(trfrac)
    ntr=round(nn*trfrac(kk));
    ind=randperm(nn); ind=ind(1:ntr)';
    ind_test=setdiff(1:prod(sz), ind);
    [I,J,K]=ind2sub(sz, ind);

    ind2=randperm(length(ind));
    ind2=ind2(1:round(length(ind2)*noisefr));

    yy=X0(ind);
    yy(ind2)=yy(ind2)+randn(length(ind2),1);

    err0=yy-X0(ind);
    
    for ii=1:length(lambda)
      for jj=1:length(methods)
        switch(methods{jj})
         case 'constraint'
          tic;
          [X,Z,Y,fval,res]=tensorconst_adm(zeros(sz),{I,J,K},yy,lambda(ii,jj));
          time(ll,ii,jj,kk)=toc;
         
         case 'matrix'
          J2=sub2ind(sz(2:3), J, K);
          tic;
          [X,Z,A,fval,res]=matrixl1_adm(zeros(sz(1), sz(2)*sz(3)),{I,J2},yy,lambda(ii,jj),'verbose',0);
          time(ll,ii,jj,kk)=toc;
          X=reshape(X, sz);
         case 'tensor'
          tic;
          [X,Z,A,beta,fval,res]=tensorl1_adm(zeros(sz),{I,J,K},yy,lambda(ii,jj),'verbose',0);
          time(ll,ii,jj,kk)=toc;
          end
        est(ll,ii,jj,kk)=norm(X0(ind)-X(ind))/norm(X0(ind));
        gen(ll,ii,jj,kk)=norm(X0(ind_test)-X(ind_test))/norm(X0(ind_test));
        memo(ll,ii,jj,kk)=struct('err0',err0,'err',yy-X(ind) ,'fval',fval,'res',res);
      end
      fprintf('tr=%g lmd=%s est=%s gen=%s\n',trfrac(kk),printvec(lambda(ii,:)), ...
              printvec(est(ll,ii,:,kk)), printvec(gen(ll,ii,:,kk)));
    end
    
  end
% $$$   lambda_mean=zeros(length(trfrac),1);
% $$$   for kk=1:length(lambda_mean)
% $$$     weight=exp(-100*gen(ll,:,jj,kk)); weight=weight/sum(weight);  
% $$$     lambda_mean(ii)=exp(log(lambda)*weight);
% $$$   end

end


if 0
  
  
  
err=yy-X0(ind); % true error vector (sparse)

[mm,ix]=min(err1);

X=memo(ix).X;

Z{1}=flatten(X,1);
Z{2}=flatten(X,2);
Z{3}=flatten(X,3);


figure
subplot(2,3,1); plot([svd(flatten(X0,1)), svd(Z{1})],'-x','linewidth',2)
grid on;
legend('True', 'Estimated')
[U0,S0,V0]=pca(flatten(X0,1),7,10);
[U,S,V]=pca(Z{1},7,10);
subplot(2,3,4); imagesc(U0'*U)

subplot(2,3,2); plot([svd(flatten(X0,2)), svd(Z{2})],'-x','linewidth',2)
grid on;
legend('True', 'Estimated')
[U0,S0,V0]=pca(flatten(X0,2),8,10);
[U,S,V]=pca(Z{2},8,10);
subplot(2,3,5); imagesc(U0'*U)

subplot(2,3,3); plot([svd(flatten(X0,3)), svd(Z{3})],'-x','linewidth',2)
grid on;
legend('True', 'Estimated')
[U0,S0,V0]=pca(flatten(X0,3),9,10);
[U,S,V]=pca(Z{3},9,10);
subplot(2,3,6); imagesc(U0'*U);

% Compute ROC curve
[ss,ix]=sort(-abs(yy-X(ind)));
for ii=1:length(ix)
tp(ii)=sum(abs(err(ix(1:ii)))>0)/length(ind2);
fp(ii)=sum(abs(err(ix(1:ii)))==0)/(ntr-length(ind2));
end

figure, plot(fp, tp, '-x', 'linewidth',2);
grid on;
xlabel('FP rate');
ylabel('TP rate');

end