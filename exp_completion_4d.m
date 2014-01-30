% exp_completion_4d - performs completion experiments on 4th order tensors
% (renamed from test_compare_4d)
%
% See also
%  exp_completion, tensorconst_subset_adm
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


nrep=50;
nsample = 1;
sz=[20 20 20 20];
trfrac=0.02:0.02:0.4;

tol=1e-4;

methods = {'matrix','subset','constraint'};
base_err = cumsum([0,1,1]);

for ll=1:nsample
  dtrue=[2,2,2,2];

  fprintf('====================== sz=%s dtrue=%s =============================\n', printvec(sz), printvec(dtrue));


  err=zeros(nrep, length(trfrac), 2);
  gval=zeros(nrep, length(trfrac), 2);

  for kk=1:nrep
    X0=randtensor(sz,dtrue);
    nn=prod(sz);

    for ii=1:length(trfrac)
      ntr=round(nn*trfrac(ii));
      ind=randperm(nn); ind=ind(1:ntr)';
      ind_test=setdiff(1:prod(sz), ind);
      [I,J,K,L]=ind2sub(sz,ind);
      yy=X0(ind);

      for mm=1:length(methods)
        switch(methods{mm})
         case 'matrix'
          %% Tensor as a matrix
          tic;
            szp=[sz(1)*sz(2) sz(3)*sz(4)];
            [Ip,Jp]=ind2sub(szp,ind);
          [X1,Z1,A1,fval1,gval1]=matrix_adm(zeros(szp), {Ip,Jp}, yy, 0, 'tol',tol);
          X1=reshape(X1,sz);
          time(kk,ii,mm)=toc;
          gval(kk,ii,mm)=max(gval1);
          err(kk,ii,base_err(mm)+1)=norm(X1(:)-X0(:));
         case 'subset'
          indUnfold = {[1 2],[3 4]; [1 3], [2 4]};
          tic;
          [X,Z,Y,fval,gvals]=tensorconst_subset_adm(zeros(sz),{I,J,K,L},yy,0,indUnfold,'tol', tol);
          time(kk,ii,mm)=toc;
          gval(kk,ii,mm)=gvals(end);
          err(kk,ii,base_err(mm)+1)=norm(X(:)-X0(:));
 
         case 'constraint'
          %% Constrained
          tic;
          [X,Z,Y,fval,gvals]=tensorconst_adm(zeros(sz),{I,J,K,L},yy,0,'tol', tol);
          time(kk,ii,mm)=toc;
          gval(kk,ii,mm)=gvals(end);
          err(kk,ii,base_err(mm)+1)=norm(X(:)-X0(:));
         case {'tucker','tuckertrue'}
          %% Tucker
          Xobs=zeros(sz);
          Xobs(ind)=X0(ind);
          Xobs(ind_test)=nan;
          Options(5)=100;
          if strcmp(methods{mm},'tuckertrue')
            dd = dtrue;
          else
            dd = round(dtrue*1.2);
          end
          tic;
          [Factors,G,ExplX,Xm]=tucker(Xobs, dd, Options);
          time(kk,ii,mm)=toc;
          gval(kk,ii,mm)=nan;
          err(kk,ii,base_err(mm)+1)=norm(Xm(:)-X0(:));
        otherwise
         error('Method [%s] unknown!', methods{mm});
        end
      end
%      fprintf('frac=%g\nerr1=[%g %g %g]  err2=%g\n',...
%              trfrac(ii), err(kk,ii,1), err(kk,ii,2), err(kk,ii,3),...
%              err(kk,ii,4));
     fprintf('frac=%g\nerr=%s\n',trfrac(ii), printvec(err(kk,ii,:)));
      fprintf('time=%s\n', printvec(time(kk,ii,:)));
      end
  end

  file_save=sprintf('result_compare_4d_%d_%d_%d_%d_%d_%d_%d_%d.mat',sz(1),sz(2),sz(3),sz(4),dtrue(1),dtrue(2),dtrue(3),dtrue(4));
%  file_save=sprintf('result_tol=1e-4_%d_%d_%d_%d_%d_%d.mat',sz(1),sz(2),sz(3),dtrue(1),dtrue(2),dtrue(3));


  save(file_save,'nrep', 'sz', 'dtrue', 'methods','err', 'trfrac','gval','time','indUnfold');
   
    
end

nm=length(methods);
figure, 
subplot(2,1,1);
h=errorbar_logsafe(trfrac'*ones(1,nm), shiftdim(mean(err)), shiftdim(std(err)));
set(h,'linewidth',2);
set(gca,'fontsize',14,'yscale','log');
ylim([1e-5 1e+2]);


grid on;
xlabel('Fraction of observed elements');
ylabel('Estimation error');
legend(methods);

subplot(2,1,2);
h=errorbar(trfrac'*ones(1,nm), shiftdim(mean(time)), ...
           shiftdim(std(time)));
set(h,'linewidth',2);
set(gca,'fontsize',14);
grid on;
yl=ylim;
ylim([0 yl(2)]);
xlabel('Fraction of observed elements');
ylabel('CPU time');
