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

% if lhs:
    // Rho is constant
    // U is constant as we restrict to only moving in y dimension
    // V is moving at 0.1
    dst[2] = (dst[2] / dst[0] - 0.1) * dst[0];
    // Change E to account for this
    dst[3] -= 0.5 * dst[0] * 0.1 * 0.1;
% else:
    // Rho is constant
    // U is constant as we restrict to only moving in y dimension
    // V is moving at 0.1
    dst[2] = (dst[2] / dst[0] + 0.1) * dst[0];
    // Change E to account for this
    dst[3] += 0.5 * dst[0] * 0.1 * 0.1;
% endif

</%pyfr:kernel>
