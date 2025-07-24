<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='siinterp' ndim='1'
              fidx='in ixdtype_t[1]'
              mat='in fpdtype_t[${str(nftps)}]'
              dst='out fpdtype_t[${str(nvars)}]'
              src='in broadcast fpdtype_t[${str(nvars)}][${str(ninterfpts)}]'>
    ixdtype_t _fidx = fidx[0];
    fpdtype_t acc = 0.0;
% for var in range(nvars):
    acc = 0.0;
    for(int i = 0; i < ${str(nftps)}; i++) {
        acc += mat[i] * src[${var}][_fidx * ${str(nftps)} + i];
    }
    dst[${var}] = acc;
% endfor
</%pyfr:kernel>
