<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    fpdtype_t p_i = ${c['gamma'] - 1.0}*(ul[${nvars - 1}]
                  - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))});
    fpdtype_t u2_i = (1.0 / (ul[0] * ul[0]))*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
    fpdtype_t c_i = sqrt(${c['gamma']} * p_i / ul[0]);
    fpdtype_t ht_i = c_i * c_i / ${c['gamma'] - 1.0} + 0.5 * u2_i;
    fpdtype_t rplus_i = - sqrt(u2_i) + 2.0 * c_i / ${c['gamma'] - 1.0};
    
    // Quad
    fpdtype_t a_quad = 2.0 + 4.0 / ${c['gamma'] - 1.0};
    fpdtype_t b_quad = -4.0 * rplus_i;
    fpdtype_t c_quad = ${c['gamma'] - 1.0} * (rplus_i * rplus_i - 2.0 * ht_i);
    fpdtype_t c_b_0 = - b_quad / (2.0 * a_quad) + sqrt(b_quad * b_quad - 4.0 * a_quad * c_quad) / (2.0 * a_quad);
    fpdtype_t c_b_1 = - b_quad / (2.0 * a_quad) - sqrt(b_quad * b_quad - 4.0 * a_quad * c_quad) / (2.0 * a_quad);
    fpdtype_t c_b = fmax(c_b_0, c_b_1);

    fpdtype_t u_b = 2.0 * c_b / ${c['gamma'] - 1.0} - rplus_i;
    fpdtype_t m_b = u_b / c_b;
    fpdtype_t p_b = ${c['pt']} * pow(1.0 + 0.5 * ${c['gamma'] - 1.0} * m_b * m_b, -${c['gamma'] / (c['gamma'] - 1.0)});
    fpdtype_t tcp_b = ${c['cpTt']} / (1.0 + 0.5 * ${c['gamma'] - 1.0} * m_b * m_b);

    ur[0] = (p_b / tcp_b) * ${1.0/c['Rdcp']};
% for i, v in enumerate(c['vc']):
    ur[${i + 1}] = ${v}*ur[0]*u_b;
% endfor
    ur[${nvars - 1}] = ${c['cpTt']} * ur[0] - p_b;
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
