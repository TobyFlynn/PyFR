<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<% gmo = c['gamma'] - 1.0 %>
<% gamma = c['gamma'] %>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t, var_p'>
    fpdtype_t inv = 1.0/ul[0];
    fpdtype_t p_b = var_p;
    fpdtype_t p_i = ${gmo}*ul[${nvars - 1}]
                  - ${0.5*gmo}*inv*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
    fpdtype_t a2_i = ${gamma} * p_i * inv;
    fpdtype_t rho_b = ${gamma} * p_b / a2_i;

    ur[0] = rho_b;
% for i in range(ndims):
    ur[${i + 1}] = rho_b*(ul[${i + 1}]*inv);
% endfor
    ur[${nvars - 1}] = p_b*${1.0/gmo}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>
<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>