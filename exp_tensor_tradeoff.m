% exp_tensor_tradeoff - runs an experiment for CP tensor
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


RandStream.setGlobalStream ...
         (RandStream('mt19937ar','seed',sum(100*clock)));

gitstring = gitlog;

nrep=10;
nsample = 1;
sz=[10 10 5];
trfrac=0.05:0.05:0.95;
sigma=0.001;
tol=1e-3;

% methods = {'matrix','constraint','nn','parafac','parafactrue'}; % , 'l2ball'};
methods = {'constraint','l2ball'};


methodParameters.gamma=0.001;                 % Parameter that ponders the importance of the regularizer. It can take any positive real value.
methodParameters.beta=0.1;                  % Parameter of ADMM (see eq. 9 in the paper). It can take any positive real value.
% methodParameters.radius=1;                % radius of the \ell_2 ball (see Sec. 3 of the paper). In principle it can take any positive real value.
                                            % If it is not specified, it is estimated using last formula in pag. 6 in the paper.
methodParameters.nIt=200;                   
methodParameters.threshold=10^-20;

mstr=method_names_string(methods);

for ll=1:nsample
  % dtrue=round(rand(1,3)*40);
  dtrue=5;
  % dtrue=[50 50 5];
  for kk=1:nrep
    X0=randtensor3(sz,dtrue);
    nn=prod(sz);

file_save=sprintf('result_tensor_tradeoff_%d_%d_%d_%d_%s_nrep=%d_sigma=%g_tol=%g.mat',sz(1),sz(2),sz(3),dtrue,mstr,nrep,sigma,tol);

    for ii=1:length(trfrac)
      ntr=round(nn*trfrac(ii));
      ind=randperm(nn); ind=ind(1:ntr)';
      ind_test=setdiff(1:prod(sz), ind);
      [I,J,K]=ind2sub(sz,ind);
      yy=X0(ind)+sigma*randn(length(ind),1);

      for mm=1:length(methods)
        clear X res
        switch(methods{mm})
         case 'matrix'
          %% Tensor as a matrix
          J1=sub2ind(sz(2:end), J, K);
          t0=cputime;
          [X,Z1,Y1,fval1,res]=matrix_adm(zeros(sz(1), prod(sz(2:end))),{I,J1}, yy, 0, 'tol', tol);
          time(kk,ii,mm)=cputime-t0;
         case 'constraint'
          %% Constrained
          t0=cputime;
          [X,Z,Y,fval,res]=tensorconst_adm(zeros(sz),{I,J,K},yy,0,'tol',tol);
          time(kk,ii,mm)=cputime-t0;
         case 'mixture'
          %% Mixture
          t0=cputime;
          [X,Z2,fval2,res]=tensormix_adm(zeros(sz), {I,J,K}, yy, ...
                                         0, 'tol', tol);
          time(kk,ii,mm)=cputime-t0;
         case 'nn'
          t0=cputime;
          [X,Z,Y,fval,res]=tensorcomplnn_adm(X0, {I,J,K}, yy, ...
                                             0, 0, 'tol', tol, ...
                                             'display', 1,...
                                             'maxiter', 30);
          time(kk,ii,mm)=cputime-t0;
         case {'parafac', 'parafactrue'}
          %% PARAFAC
          yfact=std(yy)*2;
          Xobs=zeros(sz);
          Xobs(ind)=yy/yfact;
          Xobs(ind_test)=nan;
          Options(5)=100;
          if strcmp(methods{mm},'parafactrue')
            dd = dtrue;
          else
            dd = min(round(dtrue*1.2), max(sz));
          end
          t0=cputime;
          Factors=parafac(Xobs, dd, Options);
          X=nmodel(Factors)*yfact;
          time(kk,ii,mm)=cputime-t0;
          res=nan;
         case 'l2ball'
          Knowns=zeros(sz);
          Knowns(ind)=1;
          Xobs=zeros(sz);
          Xobs(ind)=(yy-mean(yy))/std(yy);
          t0=cputime;
          data=completion2MTL(struct('Tensor', tensor(Xobs), 'KnownInputs', tensor(Knowns)),[]);
          l2Ball=MLMTL_ConvexL2BallRadiusMTL(methodParameters, 'l_2 Ball');
          l2Ball=train(l2Ball, data);
          time(kk,ii,mm)=cputime-t0;
          X=l2Ball.model.allW*std(yy)+mean(yy);
          res=nan;
         otherwise
          error('Method [%s] unknown!', methods{mm});
        end
        res(kk,ii,mm)=res(end);
        err(kk,ii,mm)=norm(X(ind_test)-X0(ind_test))/norm(X0(ind_test));
      end
     fprintf('frac=%g\nerr=%s\n',trfrac(ii), printvec(err(kk,ii,:)));
      fprintf('time=%s\n', printvec(time(kk,ii,:)));
%     fprintf('frac=%g\nerr1=%s  err2=%g  err3=%g  err4=%g err5=%g\n',...
%              trfrac(ii), printvec(err(kk,ii,1:3)),...
%              err(kk,ii,4), err(kk,ii,5), err(kk,ii,6), err(kk,ii,7));
%      fprintf('time1=%g      time2=%g time3=%g time4=%g time5=%g\n', time(kk,ii,1),time(kk,ii,2),time(kk,ii,3),time(kk,ii,4),time(kk,ii,5));
      end
  end



  save(file_save,'nrep', 'sz', 'dtrue', 'sigma', 'methods','err', 'trfrac','res','time','gitstring');
   
    
end

nm=length(methods);
figure
subplot(1,2,1);
h=errorbar(trfrac'*ones(1,nm), shiftdim(mean(err)), ...
           shiftdim(std(err)));
set(h, 'linewidth', 2);
set(gca,'fontsize', 14);
xlabel('Fraction of observed elements');
ylabel('Generalization error');
legend(methods);
grid on;

subplot(1,2,2);
h=errorbar(trfrac'*ones(1,nm), shiftdim(mean(time)), ...
           shiftdim(std(time)));
set(h, 'linewidth', 2);
set(gca,'fontsize', 14);
xlabel('Fraction of observed elements');
ylabel('CPU time');
legend(methods);
grid on;
