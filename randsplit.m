% randsplit - randomly split indices into training and testing
%
% Syntax
%  [ind_tr, ind_te]=randsplit(n, trfrac)
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [ind_tr, ind_te]=randsplit(n, trfrac)

ind=randperm(n)';

ntr=round(n*trfrac);

ind_tr=ind(1:ntr);
ind_te=ind(ntr+1:end);