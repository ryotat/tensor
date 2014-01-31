% completion2MTL - this is a simplified version of completion2MTL
%                  written by Bernardino Romera-Paredes
%
% Example
%  addpath tensor_toolbox % path to SANDIA tensor toolbox 
%  sz=[50 50 20]; rr=[7 8 9];
%  X0=randtensor3(sz, rr);
%  [ind_tr, ind_te]=randsplit(prod(sz), 0.1);
%  Knowns=zeros(sz);
%  Knowns(ind_tr)=1;
%  Xobs=zeros(sz);
%  Xobs(ind_tr)=X0(ind_tr);
%  data=completion2MTL(struct('Tensor', tensor(Xobs), 'KnownInputs', tensor(Knowns)),[]);
%
% See also
%  tensor, randsplit
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [dataConf DataParameters] = completion2MTL(dataConf, DataParameters)
%COMPLETION2MTL Summary of this function goes here
%   Detailed explanation goes here

noisyWTensor=dataConf.Tensor;
KnownInputs=dataConf.KnownInputs;

dimensions=size(noisyWTensor);
nAttrs=dimensions(1);
nTasks=prod(dimensions(2:end));
noisyW=double(reshape(noisyWTensor, [nAttrs, nTasks]));
knowns=double(reshape(KnownInputs, [nAttrs, nTasks]));

trainYCell=cell(1,nTasks);
trainXCell=cell(1,nTasks);
testYCell=cell(1,nTasks);
testXCell=cell(1,nTasks);

for i=1:nTasks
    knownT=knowns(:,i);
    present=find(knownT);
    nInstances=length(present);
    X=zeros(nAttrs, nInstances);
    Y=zeros(nInstances, 1);
    for j=1:length(present)
        X(present(j), j)=1;
        Y(j)=noisyW(present(j),i);
    end
    
    present=find(knownT==0);
    nInstances=length(present);
    XTest=zeros(nAttrs, nInstances);
    YTest=zeros(nInstances, 1);
%    for j=1:length(present)
%        XTest(present(j), j)=1;
%        YTest(j)=W(present(j),i);
%    end
    
%     XTest=eye(nAttrs);
%     YTest=W(:,i);
    
    trainYCell{i}=Y;
    trainXCell{i}=X;
    testYCell{i}=YTest;
    testXCell{i}=XTest;
end

dataConf.trainXCell=trainXCell;
dataConf.trainYCell=trainYCell;
dataConf.testXCell=testXCell;
dataConf.testYCell=testYCell;
% dataConf.validation_testXCell=validationXCell;
% dataConf.validation_testYCell=validationYCell;
%dataConf.W=W;
%dataConf.WTensor=WTensor;
dataConf.indicators=dimensions;

end
