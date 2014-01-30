% plot_threshold_vs_normalized_rank - plots phase transition threshold 
%                        for tensor completion against normalized rank
%
% Syntax
%  [rst, frac, dim]=plot_threshold_vs_normalized_rank(files,tol)
%
% See also
%  exp_completion, plot_nips2011
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

function [rst, frac, dim]=plot_threshold_vs_normalized_rank(files,tol)

if ~exist('tol','var')
  tol=0.01;
end


ns=length(files);

frac=zeros(ns,1);

for ii=1:ns
  S=load(files{ii});
  sz=S.sz;
  rs(ii,:)=S.dtrue;
  ix=min(find(mean(S.err)<tol));
  if isempty(ix)
    frac(ii)=nan;
  else
    frac(ii)=S.trfrac(ix);
  end
  

  nrep=10;
%  nmtmp=0;
%  for jj=1:nrep
%    nmtmp=nmtmp+shattennorm(randtensor3(sz,S.dtrue));
%  end
%  nm(ii)=nmtmp/nrep;
end
% $$$ 
% $$$ figure;
% $$$ plot(sum(rs,2),frac,'x','linewidth',2);
% $$$ for ii=1:ns
% $$$   text(sum(rs(ii,:)),frac(ii),...
% $$$        sprintf('[%d %d %d]',rs(ii,1),rs(ii,2),rs(ii,3)));
% $$$ end

if size(rs,2)>1
  rst=zeros(size(rs));
  for kk=1:size(rs,2)
    rst(:,kk)=min(rs(:,kk), prod(rs,2)./rs(:,kk));
  end
  if size(rs,2)==3
    rst_check=[min(rs(:,1), rs(:,2).*rs(:,3)),...
               min(rs(:,2), rs(:,1).*rs(:,3)),...
               min(rs(:,3), rs(:,1).*rs(:,2))];
    if ~isequal(rst,rst_check)
      error('rst check for 3rd order tensor failed');
    end
  end
else
  rst=[rs,rs];
end


% dim=rs*[50 50 20]'+rs(:,1).*rs(:,2).*rs(:,3) -sum(rs.^2,2);
%dim=max(rst,[],2);%sum(rst,2);
% dim=sum(rst,2);
% dim=sum(sqrt(rst./(ones(ns,1)*sz)),2).^2/(size(sz,2)^2*size(rst,2)^2);
dim=(sum(sqrt(1./sz))*sum(sqrt(rst),2)).^2/(size(sz,2)^2*size(rst,2)^2);

%dim=nm';
% figure;
plot(dim,frac,'x','linewidth',2);

if 0 % size(rs,2)>1
for ii=1:ns
  text(dim(ii),frac(ii),...
       sprintf('[%d %d %d]',rs(ii,1),rs(ii,2),rs(ii,3)));
end
end
ix=find(~isnan(frac));
p=polyfit(dim(ix),frac(ix),1)
hold on;
plot(xlim,polyval(p,xlim),'--','color',[.5 .5 .5],'linewidth', 2)
h=get(gca,'children');
set(gca,'children',h([2:end,1]));




%dim = prod(rs,2)+sum(rs.*(ones(ns,1)*sz),2)-sum(rs.^2,2);
%x = dim.^(1/3);
% $$$ dim(:,1) = max([rst(:,1)*(sz(1)+prod(sz(2:3)))   - rst(:,1).^2,...
% $$$            rst(:,2)*(sz(2)+prod(sz([1,3]))) - rst(:,2).^2,...
% $$$           rst(:,3)*(sz(3)+prod(sz(1:2)))   - rst(:,3).^2],[],2);
% $$$ 
% $$$ dim(:,2) = rst(:,1)*sz(1)+rst(:,2)*sz(2)+rst(:,3)*sz(3); +rst(:,1).*rst(:,2).*rst(:,3)-rst(:,1).^2-rst(:,2).^2-rst(:,3).^2;
% $$$ 
% $$$ 
% $$$ dim=min(dim,[],2);
% $$$ 
% $$$ figure, plot(dim, frac,'x', 'linewidth',2)
% $$$ for ii=1:ns
% $$$   text(dim(ii),frac(ii),...
% $$$        sprintf('[%d %d %d]',rs(ii,1),rs(ii,2),rs(ii,3)));
% $$$ end
