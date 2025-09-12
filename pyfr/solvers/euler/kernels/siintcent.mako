<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='siintcent' ndim='1'
              entmin_lhs='inout view fpdtype_t'>
    // entmin_lhs = fmin(entmin_lhs, entmin_rhs);
    entmin_lhs = entmin_lhs;
</%pyfr:kernel>
