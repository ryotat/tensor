function str=method_names_string(methods, len)

if ~exist('len','var')
  len=3;
end


str='';
for mm=1:length(methods)
 method=methods{mm};
  str = [str, method(1:min(len,length(method)))];
  if mm<length(methods)
    str = [str, '_'];
  end
end
