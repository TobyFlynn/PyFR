<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='sicopy' ndim='1'
              src='in view fpdtype_t[${str(nvars)}]'
              dst='out fpdtype_t[${str(nvars)}]'>
% for i in range(nvars):
    dst[${i}] = src[${i}];
% endfor
</%pyfr:kernel>
