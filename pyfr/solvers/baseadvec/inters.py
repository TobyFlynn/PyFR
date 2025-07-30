import itertools as it
from functools import cached_property
import math
import numpy as np

from pyfr.nputil import npeval
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.shapes import BaseShape
from pyfr.solvers.base import BaseInters
from pyfr.solvers.base.inters import _get_inter_objs


class BaseAdvectionIntersMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ef_enabled = (self.cfg.get('solver', 'shock-capturing') ==
                            'entropy-filter' and
                            self.cfg.getint('solver', 'order'))


class BaseAdvectionIntInters(BaseAdvectionIntersMixin, BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)

        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')

        # Generate the additional view matrices for entropy filtering
        if self._ef_enabled:
            self._entmin_lhs = self._view(
                lhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
            self._entmin_rhs = self._view(
                rhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
        else:
            self._entmin_lhs = self._entmin_rhs = None

        # Generate the constant matrices
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = self._get_perm_for_view(lhs, 'get_scal_fpts_for_inter')


class BaseAdvectionMPIInters(BaseAdvectionIntersMixin, BaseInters):
    # Starting tag used for MPI
    BASE_MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self._rhsrank = rhsrank

        # Name our interface so we can match kernels to MPI requests
        self.name = f'p{rhsrank}'

        # MPI request tag counter
        self._mpi_tag_counter = it.count(self.BASE_MPI_TAG)

        # Generate the left hand view matrix and its dual
        self._scal_lhs = self._scal_xchg_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = be.xchg_matrix_for_view(self._scal_lhs)

        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

        # Kernels
        self.kernels['scal_fpts_pack'] = lambda: be.kernel(
            'pack', self._scal_lhs
        )
        self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
            'unpack', self._scal_rhs
        )

        # Associated MPI requests
        scal_fpts_tag = next(self._mpi_tag_counter)
        self.mpireqs['scal_fpts_send'] = lambda: self._scal_lhs.sendreq(
            self._rhsrank, scal_fpts_tag
        )
        self.mpireqs['scal_fpts_recv'] = lambda: self._scal_rhs.recvreq(
            self._rhsrank, scal_fpts_tag
        )

        if self._ef_enabled:
            self._entmin_lhs = self._xchg_view(
                lhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
            self._entmin_rhs = be.xchg_matrix_for_view(self._entmin_lhs)

            self.kernels['ent_fpts_pack'] = lambda: be.kernel(
                'pack', self._entmin_lhs
            )
            self.kernels['ent_fpts_unpack'] = lambda: be.kernel(
                'unpack', self._entmin_rhs
            )

            ent_fpts_tag = next(self._mpi_tag_counter)
            self.mpireqs['ent_fpts_send'] = lambda: self._entmin_lhs.sendreq(
                self._rhsrank, ent_fpts_tag
            )
            self.mpireqs['ent_fpts_recv'] = lambda: self._entmin_rhs.recvreq(
                self._rhsrank, ent_fpts_tag
            )
        else:
            self._entmin_lhs = self._entmin_rhs = None


class BaseAdvectionBCInters(BaseAdvectionIntersMixin, BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfg)

        self.cfgsect = cfgsect
        self.bccomm = bccomm
        self.name = cfgsect.removeprefix('soln-bcs-')

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = self._get_perm_for_view(lhs, 'get_scal_fpts_for_inter')

        # LHS view and constant matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

        # Make the simulation time available inside kernels
        self._set_external('t', 'scalar fpdtype_t')

        if self._ef_enabled:
            self._entmin_lhs = self._view(lhs, 'get_entmin_bc_fpts_for_inter')
        else:
            self._entmin_lhs = None

    @classmethod
    def preparefn(cls, bciface, mesh, elemap):
        pass

    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self.cfg.items_as('constants', float)

        cfg, sect = self.cfg, self.cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.getexpr(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.getexpr(sect, k), cc) for k in opts]

    def _exp_opts(self, opts, lhs, default={}):
        cfg, sect = self.cfg, self.cfgsect

        subs = cfg.items('constants')
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=str(math.pi))

        exprs = {}
        for k in opts:
            if k in default:
                exprs[k] = cfg.getexpr(sect, k, default[k], subs=subs)
            else:
                exprs[k] = cfg.getexpr(sect, k, subs=subs)

        if (any('ploc' in ex for ex in exprs.values()) and
            'ploc' not in self._external_args):
            spec = f'in fpdtype_t[{self.ndims}]'
            value = self._const_mat(lhs, 'get_ploc_for_inter')

            self._set_external('ploc', spec, value=value)

        return exprs

# Current assumptions:
# - 2D
# - Single rank
# - The interface is parallel to the y-axis (i.e. sign of the normal's x-component can split interface, also for pt to face mapping)
# - Linear interface
class BaseAdvectionSlidingInters(BaseAdvectionIntersMixin, BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfg)

        self.cfgsect = cfgsect
        self.name = cfgsect.removeprefix('soln-sliding-interface-')
        self.order = cfg.getint('solver', 'order')

        # Get translation
        self.ul = cfg.getfloat(cfgsect, 'ul')
        self.vl = cfg.getfloat(cfgsect, 'vl')
        self.ur = cfg.getfloat(cfgsect, 'ur')
        self.vr = cfg.getfloat(cfgsect, 'vr')

        # Set functions for straight line interface
        self._check_pt_in_face = self._check_pt_in_line_face
        self.etype = lhs[0][0]
        linepts = cfg.get('solver-interfaces-line', 'flux-pts')
        self.fpts = get_quadrule('line', linepts, qdeg=self.order+2).pts

        # Split into lhs and rhs of the sliding interface
        self.lhs, self.rhs = self._split_lhs_rhs(elemap, lhs)

        # Set to correct values, parent sets these before split
        self.ninters = len(self.lhs)
        self.ninterfpts = sum(elemap[etype].nfacefpts[fidx]
                              for etype, eidx, fidx in self.lhs)

        self._set_original_fpts_and_bounds(self.lhs, self.rhs)

        # View and constant matrices
        self._scal_lhs = self._scal_view(self.lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = self._scal_view(self.rhs, 'get_scal_fpts_for_inter')
        self._pnorm_lhs = self._const_mat(self.lhs, 'get_pnorms_for_inter')
        self._pnorm_rhs = self._const_mat(self.rhs, 'get_pnorms_for_inter')

        # Copies of face point data
        tags = {'align'}
        mat_size = (self.nvars, self.ninterfpts)
        zero_init = np.full(mat_size, 0.0)
        self._scal_lhs_copy = self._be.matrix(mat_size,
                                              tags=tags, extent=f'sliding_lhs_copy_{self.name}',
                                              initval=zero_init)
        self._scal_rhs_copy = self._be.matrix(mat_size,
                                              tags=tags, extent=f'sliding_rhs_copy_{self.name}',
                                              initval=zero_init)
        self._scal_lhs_interp = self._be.matrix(mat_size,
                                              tags=tags, extent=f'sliding_lhs_interp_{self.name}',
                                              initval=zero_init)
        self._scal_rhs_interp = self._be.matrix(mat_size,
                                              tags=tags, extent=f'sliding_rhs_interp_{self.name}',
                                              initval=zero_init)

        # Make the simulation time available inside kernels
        self._set_external('t', 'scalar fpdtype_t')

        if self._ef_enabled:
            self._entmin_lhs = self._view(self.lhs, 'get_entmin_bc_fpts_for_inter')
            self._entmin_rhs = self._view(self.rhs, 'get_entmin_bc_fpts_for_inter')
        else:
            self._entmin_lhs = None
            self._entmin_rhs = None

        # Matrices for interpolation
        mat_size_fidx = (1, self.ninterfpts)
        zero_init = np.full(mat_size_fidx, 0, dtype=self._be.ixdtype)
        self._lhs_fidx = self._be.matrix(mat_size_fidx, tags=tags, 
                                         extent=f'sliding_lhs_fidx_{self.name}',
                                         initval=zero_init, dtype=self._be.ixdtype)
        self._rhs_fidx = self._be.matrix(mat_size_fidx, tags=tags, 
                                         extent=f'sliding_rhs_fidx_{self.name}',
                                         initval=zero_init, dtype=self._be.ixdtype)
        mat_size_interp = (len(self.fpts), self.ninterfpts)
        zero_init = np.full(mat_size_interp, 0.0)
        self._lhs_interp_mats = self._be.matrix(mat_size_interp, tags=tags, 
                                         extent=f'sliding_lhs_interp_mats_{self.name}',
                                         initval=zero_init)
        self._rhs_interp_mats = self._be.matrix(mat_size_interp, tags=tags, 
                                         extent=f'sliding_rhs_interp_mats_{self.name}',
                                         initval=zero_init)
        
        # Kernels common across all solver
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.sicopy')
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.siinterp')

        tplargs = dict(nvars=self.nvars)

        self.kernels['copy_fpts_lhs'] = lambda: self._be.kernel(
            'sicopy', tplargs=tplargs, dims=[self.ninterfpts], 
            src=self._scal_lhs, dst=self._scal_lhs_copy
        )
        self.kernels['copy_fpts_rhs'] = lambda: self._be.kernel(
            'sicopy', tplargs=tplargs, dims=[self.ninterfpts], 
            src=self._scal_rhs, dst=self._scal_rhs_copy
        )

        self.kernels['interp_fpts_lhs'] = lambda: self._be.kernel(
            'siinterp', tplargs=self._tplargs | dict(lhs=True), dims=[self.ninterfpts],
            src=self._scal_rhs_copy, fidx=self._lhs_fidx, mat=self._lhs_interp_mats,
            dst=self._scal_lhs_interp
        )
        self.kernels['interp_fpts_rhs'] = lambda: self._be.kernel(
            'siinterp', tplargs=self._tplargs | dict(lhs=False), dims=[self.ninterfpts],
            src=self._scal_lhs_copy, fidx=self._rhs_fidx, mat=self._rhs_interp_mats,
            dst=self._scal_rhs_interp
        )
    
    def _split_lhs_rhs(self, elemap, allf):
        # Get the normal of each face
        norm = _get_inter_objs(allf, 'get_pnorms_for_inter', self.elemap)
        norm = np.concatenate(norm)
        norm = np.atleast_2d(norm.T)

        # Currently split on the x component of the normal
        lhs = []
        rhs = []
        self.nfptstotal = 0
        for i in range(0, len(allf)):
            if norm[0][self.nfptstotal] < 0.0:
                lhs.append(allf[i])
            else:
                rhs.append(allf[i])
            etype, eidx, fidx = allf[i]
            self.nfptstotal += elemap[etype].nfacefpts[fidx]
        
        return lhs, rhs

    @cached_property
    def _face_polybasis(self):
        return get_polybasis('line', self.order + 1, self.fpts)

    def _set_original_fpts_and_bounds(self, lhs, rhs):
        self._lhs_plocs = _get_inter_objs(lhs, 'get_plocs_for_inter', self.elemap)
        self._rhs_plocs = _get_inter_objs(rhs, 'get_plocs_for_inter', self.elemap)

        op = self._face_polybasis.nodal_basis_at([-1.0, 1.0])

        self._lhs_face_bounds = []
        for _plocs in self._lhs_plocs:
            _y = [_pt[1] for _pt in _plocs]
            self._lhs_face_bounds.append(op @ _y)
        
        self._rhs_face_bounds = []
        for _plocs in self._rhs_plocs:
            _y = [_pt[1] for _pt in _plocs]
            self._rhs_face_bounds.append(op @ _y)
        
        self._lhs_plocs = np.reshape(self._lhs_plocs, (-1, 2))
        self._rhs_plocs = np.reshape(self._rhs_plocs, (-1, 2))

    def _apply_transform(self, t):
        # Apply the transform to each side of the equation
        # Only transform the points, not the face bounds 
        # (so need to account for this in the transform)
        dist_u_l = 0.0 # (self.ul - self.ur) * t
        dist_v_l = (self.vl - self.vr) * t
        dist_u_r = 0.0 # (self.ur - self.ul) * t
        dist_v_r = (self.vr - self.vl) * t
        t_ploc_lhs = np.array([[_u + dist_u_l, _v + dist_v_l] for _u, _v in self._lhs_plocs])
        t_ploc_rhs = np.array([[_u + dist_u_r, _v + dist_v_r] for _u, _v in self._rhs_plocs])

        # Mod each ploc point to match the otherside's face bounds
        lhs_min = min(np.reshape(self._lhs_face_bounds, (-1)))
        lhs_max = max(np.reshape(self._lhs_face_bounds, (-1)))
        lhs_len = lhs_max - lhs_min
        rhs_min = min(np.reshape(self._rhs_face_bounds, (-1)))
        rhs_max = max(np.reshape(self._rhs_face_bounds, (-1)))
        rhs_len = rhs_max - rhs_min
        for _ploc in t_ploc_lhs:
            if _ploc[1] < rhs_min:
                _ploc[1] += np.floor((rhs_max - _ploc[1]) / rhs_len) * rhs_len
            if _ploc[1] > rhs_max:
                _ploc[1] -= np.floor((_ploc[1] - rhs_min) / rhs_len) * rhs_len
        
        for _ploc in t_ploc_rhs:
            if _ploc[1] < lhs_min:
                _ploc[1] += np.floor((lhs_max - _ploc[1]) / lhs_len) * lhs_len
            if _ploc[1] > lhs_max:
                _ploc[1] -= np.floor((_ploc[1] - lhs_min) / lhs_len) * lhs_len
        
        return t_ploc_lhs, t_ploc_rhs
    
    def _check_pt_in_line_face(self, pt, fbounds):
        return pt[1] + 1e-10 >= fbounds[0] and pt[1] - 1e-10 <= fbounds[1]

    # Brute force search for now
    def _get_fidx_for_pts(self, pts_plocs, face_bounds):
        fidx = []
        for _ploc in pts_plocs:
            _fidx = -1
            for i in range(0, len(face_bounds)):
                if self._check_pt_in_face(_ploc, face_bounds[i]):
                    _fidx = i
                    break
            if _fidx == -1:
                raise Exception(f'A sliding interface point ({_ploc[0]},{_ploc[1]}) is not within any faces')
            fidx.append(_fidx)
        return fidx

    # Assume y axis aligned line
    def _get_rloc(self, pt_ploc, face_bounds):
        return 2.0 * ((pt_ploc[1] - face_bounds[0]) / (face_bounds[1] - face_bounds[0])) - 1.0

    def _get_interp_mats_for_pts(self, pts_ploc, faces_bounds, fidxs):
        mats = []
        for _pt_ploc, _fidx in zip(pts_ploc, fidxs):
            face_bounds = faces_bounds[_fidx]
            rloc = self._get_rloc(_pt_ploc, face_bounds)
            mats.append(self._face_polybasis.nodal_basis_at([rloc]))
        return mats

    def interpolate(self, t):
        # Get the current plocs of each face point
        lhs_plocs, rhs_plocs = self._apply_transform(t)
        lhs_face_bounds, rhs_face_bounds = self._lhs_face_bounds, self._rhs_face_bounds

        # Work out which face contains each face point
        lhs_pts_rhs_fidx = self._get_fidx_for_pts(lhs_plocs, rhs_face_bounds)
        rhs_pts_lhs_fidx = self._get_fidx_for_pts(rhs_plocs, lhs_face_bounds)

        # Calculate matrix to interpolate to face point
        lhs_pts_interp_matrices = self._get_interp_mats_for_pts(lhs_plocs, rhs_face_bounds, lhs_pts_rhs_fidx)
        rhs_pts_interp_matrices = self._get_interp_mats_for_pts(rhs_plocs, lhs_face_bounds, rhs_pts_lhs_fidx)

        # Set backend matrices
        self._lhs_fidx.set(np.reshape(lhs_pts_rhs_fidx, (-1, 1)).swapaxes(0,1))
        self._rhs_fidx.set(np.reshape(rhs_pts_lhs_fidx, (-1, 1)).swapaxes(0,1))
        self._lhs_interp_mats.set(np.reshape(lhs_pts_interp_matrices, (-1, len(self.fpts))).swapaxes(0,1))
        self._rhs_interp_mats.set(np.reshape(rhs_pts_interp_matrices, (-1, len(self.fpts))).swapaxes(0,1))

