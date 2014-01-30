% printvec - prints a vector into a string
%
% Syntax
%  str=printvec(vv,ml)
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function str=printvec(vv,ml)

if ~exist('ml','var')
  ml=10;
end


vv=vv(:);

str = '[';

for ii=1:min(length(vv),ml)-1
  str = [str, sprintf('%g ', vv(ii))];
end

if length(vv)>ml
  str = [str, '...'];
end

str = [str, sprintf('%g]',vv(end))];
