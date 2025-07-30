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

    // Account for changing the frame of reference
% if lhs:
    // Rho is constant
    // Rhou
    dst[1] = (dst[1] / dst[0] + ${vel_r[0] - vel_l[0]}) * dst[0];
    // Rhov
    dst[2] = (dst[2] / dst[0] + ${vel_r[1] - vel_l[1]}) * dst[0];
    // E
    dst[3] += 0.5 * dst[0] * (${vel_l[0]**2} - ${vel_r[0]**2});
% else:
    // Rho is constant
    // Rhou
    dst[1] = (dst[1] / dst[0] + (${vel_l[0] - vel_r[0]})) * dst[0];
    // Rhov
    dst[2] = (dst[2] / dst[0] + (${vel_l[1] - vel_r[1]})) * dst[0];
    // E
    dst[3] += 0.5 * dst[0] * (${vel_r[0]**2} - ${vel_l[0]**2});
% endif

</%pyfr:kernel>
