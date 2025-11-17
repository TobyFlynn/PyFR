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
    fpdtype_t _u = dst[1] / dst[0];
    fpdtype_t _v = dst[2] / dst[0];
% if lhs:
    fpdtype_t new_u = _u + ${vel_r[0] - vel_l[0]};
    fpdtype_t new_v = _v + ${vel_r[1] - vel_l[1]};
% else:
    fpdtype_t new_u = _u + ${vel_l[0] - vel_r[0]};
    fpdtype_t new_v = _v + ${vel_l[1] - vel_r[1]};
% endif
% if ndims == 3:
    pdtype_t _w = dst[3] / dst[0];
% if lhs:
    fpdtype_t new_w = _w + ${vel_r[2] - vel_l[2]};
% else:
    fpdtype_t new_w = _w + ${vel_l[2] - vel_r[2]};
% endif
% endif
    // Rho is constant
    // Rhou
    dst[1] = new_u * dst[0];
    // Rhov
    dst[2] = new_v * dst[0];
% if ndims == 2:
    // E
    dst[3] -= 0.5 * dst[0] * (_u * _u - new_u * new_u + _v * _v - new_v * new_v);
% else:
    // Rhow
    dst[3] = new_w * dst[0];
    // E
    dst[4] -= 0.5 * dst[0] * (_u * _u - new_u * new_u + _v * _v - new_v * new_v + _w * _w - new_w * new_w);
% endif

</%pyfr:kernel>
