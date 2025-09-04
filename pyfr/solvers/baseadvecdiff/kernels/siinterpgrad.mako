<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='siinterpgrad' ndim='1'
              fidx='in ixdtype_t[1]'
              mat='in fpdtype_t[${str(nftps)}]'
              dst='out fpdtype_t[${str(ndims)}][${str(nvars)}]'
              src='in broadcast fpdtype_t[${str(ndims*nvars)}][${str(ninterfpts)}]'>
    ixdtype_t _fidx = fidx[0];
    fpdtype_t acc = 0.0;
% for dim in range(ndims):
% for var in range(nvars):
    acc = 0.0;
    for(int i = 0; i < ${str(nftps)}; i++) {
        acc += mat[i] * src[${dim * ndims + var}][_fidx * ${str(nftps)} + i];
    }
    dst[${dim}][${var}] = acc;
% endfor
% endfor

</%pyfr:kernel>
