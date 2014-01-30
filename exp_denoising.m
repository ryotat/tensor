% exp_denoising - performs denoising experiments
%
% See also
%  exp_completion, plot_denoising, plot_compare_denoising
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


RandStream.setDefaultStream ...
         (RandStream('mt19937ar','seed',sum(100*clock)));


gitstring = gitlog;


nsample=20;
nrep=10;

% sz=[50 50 20];
sz=input('size=');
sigma=input('sigma=');

tol=1e-4;

methods = {'constraint','mixture'}; % ,'tuckertrue'};

lambda=exp(linspace(log(0.1),log(100),20));

time=nan*ones(nrep, length(lambda), length(methods));
res=nan*ones(nrep, length(lambda), length(methods));
err=nan*ones(nrep, length(lambda), length(methods));


for ll=1:nsample
  % dtrue=[rr(ll),rr(ll),3];
  dtrue=max(1,round(rand(1,3).*sz));

  fprintf('====================== sz=%s dtrue=%s =============================\n', printvec(sz), printvec(dtrue));

  for kk=1:nrep
    X0=randtensor3(sz,dtrue);
    nn=prod(sz);
    ind=(1:nn)';
    [I,J,K]=ind2sub(sz, ind);

    [X0out,Z0,A0,fval0,res0]=tensormix_adm(zeros(sz), {I,J,K}, X0(:), ...
                                           0, 'tol', 1e-4);
      
    rank_mix(kk,:)=[sum(svd(Z0{1})>1e-9), sum(svd(Z0{2})>1e-9), sum(svd(Z0{3})>1e-9)];
    fprintf('Latent rank=%s\n', printvec(rank_mix(kk,:)));

    Y=X0+sigma*randn(sz);

    for mm=1:length(methods)
      switch(methods{mm})
       case 'constraint'
        %% Constrained
        for jj=1:length(lambda)
          t0=cputime;
          [X,Z,A,fval,ress]=tensorconst_adm(zeros(sz),{I,J,K},Y(:),lambda(jj),'tol',tol);
          time(kk,jj,mm)=cputime-t0;
          res(kk,jj,mm)=ress(end);
          err(kk,jj,mm)=norm(X(:)-X0(:));
          rank_obtained{kk,jj,mm}=[rank(flatten(X,1),1e-9),...
                              rank(flatten(X,2),1e-9),...
                              rank(flatten(X,3),1e-9)];
        end
       case 'mixture'
        %% Mixture
        for jj=1:length(lambda)
          t0=cputime;
          [X2,Z2,fval2,res2]=tensormix_adm(zeros(sz), {I,J,K}, Y(:), ...
                                           lambda(jj), 'tol', tol);
          time(kk,jj,mm)=cputime-t0;
          res(kk,jj,mm)=res2(end);
          err(kk,jj,mm)=norm(X2(:)-X0(:));
          rank_obtained{kk,jj,mm}=[rank(Z2{1},1e-9),...
                              rank(Z2{2},1e-9),...
                              rank(Z2{3},1e-9)];
        end
        
       case {'tucker','tuckertrue'}
        %% Tucker
        yfact=std(Y(:))*2;
        Xobs=Y/yfact;
        Options(5)=100;
        if strcmp(methods{mm},'tuckertrue')
          dd = dtrue;
        else
          dd = min(round(dtrue*1.5), sz);
        end
        t0=cputime;
        [Factors,G,ExplX,Xm]=tucker(Xobs, dd, Options);
        Xm=Xm*yfact;
        time(kk,1,mm)=cputime-t0;
        res(kk,1,mm)=nan;
        err(kk,1,mm)=norm(Xm(:)-X0(:));
       otherwise
        error('Method [%s] unknown!', methods{mm});
      end
      fprintf('kk=%d: [%s] err=%s\n', kk, methods{mm}, printvec(err(kk,:,mm)));

    end
    
  end
  file_save=sprintf('result_compare_full_%d_%d_%d_%d_%d_%d_nrep=%d_sigma=%g.mat',sz(1),sz(2),sz(3),dtrue(1),dtrue(2),dtrue(3),nrep,sigma);


  save(file_save,'nrep', 'sz', 'dtrue', 'methods','lambda','err','res','time','sigma','tol','rank_mix','gitstring','rank_obtained');
   
    
end

