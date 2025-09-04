<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='sicopygrad' ndim='1'
              src='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              dst='out fpdtype_t[${str(ndims * nvars)}]'>
% for i, j in pyfr.ndrange(ndims, nvars):
    dst[${i * ndims + j}] = src[${i}][${j}];
% endfor
</%pyfr:kernel>
