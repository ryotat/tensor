% exp_completion -  performs completion experiments
% (renamed from test_compare)
%
% See also
%  exp_denoising, plot_threshold_vs_normalized_rank
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

nrep=20;
nsample = 1;
sz=[50 50 20];
trfrac=0.05:0.05:0.95;

tol=1e-3;

methods = {'matrix','constraint','mixture'}; % ,'tucker','tuckertrue'};
base_err = cumsum([0, 3, 1]); % , 1, 1]);

for ll=1:nsample
  % dtrue=round(rand(1,3)*40);
  dtrue=[7,8,9];
  % dtrue=[50 50 5];


  for kk=1:nrep
    X0=randtensor3(sz,dtrue);
    nn=prod(sz);

    for ii=1:length(trfrac)
      ntr=round(nn*trfrac(ii));
      ind=randperm(nn); ind=ind(1:ntr)';
      ind_test=setdiff(1:prod(sz), ind);
      [I,J,K]=ind2sub(sz,ind);
      yy=X0(ind);

      for mm=1:length(methods)
        switch(methods{mm})
         case 'matrix'
          %% Tensor as a matrix
          tic;
          [X1,Z1,fval1,res1]=tensor_as_matrix(zeros(sz), {I,J,K}, ...
                                               yy, 0, 'tol', tol);
          time(kk,ii,mm)=toc;
          res(kk,ii,mm)=max(res1);
          for jj=1:ndims(X0)
            Xjj = flatten_adj(Z1{jj},sz,jj);
            err(kk,ii,base_err(mm)+jj)=norm(Xjj(ind_test)-X0(ind_test))/norm(X0(ind_test));
          end
         case 'constraint'
          %% Constrained
          tic;
          [X,Z,Y,fval,ress]=tensorconst_adm(zeros(sz),{I,J,K},yy,0,'tol',tol);
          time(kk,ii,mm)=toc;
          res(kk,ii,mm)=ress(end);
          err(kk,ii,base_err(mm)+1)=norm(X(ind_test)-X0(ind_test))/norm(X0(ind_test));
         case 'mixture'
          %% Mixture
          tic;
          [X2,Z2,fval2,res2]=tensormix_adm(zeros(sz), {I,J,K}, yy, ...
                                            0, 'tol', tol);
          time(kk,ii,mm)=toc;
          res(kk,ii,mm)=res2(end);
          err(kk,ii,base_err(mm)+1)=norm(X2(ind_test)-X0(ind_test))/norm(X0(ind_test));
         case {'tucker','tuckertrue'}
          %% Tucker
          yfact=std(yy)*2;
          Xobs=zeros(sz);
          Xobs(ind)=X0(ind)/yfact;
          Xobs(ind_test)=nan;
          Options(5)=100;
          if strcmp(methods{mm},'tuckertrue')
            dd = dtrue;
          else
            dd = min(round(dtrue*1.2), sz);
          end
          tic;
          [Factors,G,ExplX,Xm]=tucker(Xobs, dd, Options);
          Xm=Xm*yfact;
          time(kk,ii,mm)=toc;
          res(kk,ii,mm)=nan;
          err(kk,ii,base_err(mm)+1)=norm(Xm(ind_test)-X0(ind_test))/norm(X0(ind_test));
        otherwise
         error('Method [%s] unknown!', methods{mm});
        end
      end
     fprintf('frac=%g\nerr=%s\n',trfrac(ii), printvec(err(kk,ii,:)));
      fprintf('time=%s\n', printvec(time(kk,ii,:)));
%     fprintf('frac=%g\nerr1=%s  err2=%g  err3=%g  err4=%g err5=%g\n',...
%              trfrac(ii), printvec(err(kk,ii,1:3)),...
%              err(kk,ii,4), err(kk,ii,5), err(kk,ii,6), err(kk,ii,7));
%      fprintf('time1=%g      time2=%g time3=%g time4=%g time5=%g\n', time(kk,ii,1),time(kk,ii,2),time(kk,ii,3),time(kk,ii,4),time(kk,ii,5));
      end
  end

  file_save=sprintf('result_compare5_%d_%d_%d_%d_%d_%d_nrep=%d_tol=%g.mat',sz(1),sz(2),sz(3),dtrue(1),dtrue(2),dtrue(3),nrep,tol);


  save(file_save,'nrep', 'sz', 'dtrue', 'methods','err', 'trfrac','res','time');
   
    
end

plot_tensorworkshop10
