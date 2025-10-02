<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='sicalcmats' ndim='1'
              rloc='in fpdtype_t'
              out='out fpdtype_t[${str(nftps)}]'
              invvdm='in broadcast fpdtype_t[${str(nftps)}][${str(nftps)}]'>
    fpdtype_t ortho_basis[${str(nftps)}];

    ortho_basis[0] = 1.0;
    ortho_basis[1] = rloc;
    for(int i = 2; i < ${str(nftps)}; i++) {
        fpdtype_t aq = 2.0*i*(2.0*i-1.0)/(2.0*i*i);
        fpdtype_t cq = 2.0*i*(i-1.0)*(i-1.0)/(i*i*(2.0*i-2.0));
        ortho_basis[i] = aq*rloc*ortho_basis[i-1] - cq*ortho_basis[i-2];
    }

    for(int i = 0; i < ${str(nftps)}; i++) {
        ortho_basis[i] = sqrt(i + 0.5)*ortho_basis[i];
    }

    // Multiply ortho_basis by invvdm
    for(int i = 0; i < ${str(nftps)}; i++) {
        out[i] = 0.0;
        for(int j = 0; j < ${str(nftps)}; j++) {
            out[i] += invvdm[i][j] * ortho_basis[j];
        }
    }
</%pyfr:kernel>