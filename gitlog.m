function str=gitlog(number,hashonly)

if ~exist('number','var')
  number=1;
end

if ~exist('hashonly','var')
  hashonly=0;
end


os=computer;

if isequal(os(1:4),'PCWIN')
  str='';
  return;
end

if isequal(os(1:3),'MAC')
  gitcmd = '/usr/local/git/bin/git';
else
  gitcmd = 'git';
end

if hashonly
  command = sprintf('%s --no-pager log -%d --format="%%h"',...
		  gitcmd, number);
else
  command = sprintf('%s --no-pager log -%d --format="%%ci %%h %%s"',...
		  gitcmd, number);
end

[res,str]=system(command);

str(end)=[];