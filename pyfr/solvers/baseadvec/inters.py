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

import time

# Current assumptions:
# - 2D
# - The interface is parallel to the y-axis (i.e. sign of the normal's x-component can split interface, also for pt to face mapping)
# - Linear interface
# - Entropy filtering is currently wrong
class BaseAdvectionSlidingInters(BaseAdvectionIntersMixin, BaseInters):
    type = None

    LHS_MPI_TAG = 3456
    RHS_MPI_TAG = 3457

    def __init__(self, be, lhs, elemap, cfgsect, cfg, sicomm):
        super().__init__(be, lhs, elemap, cfg)

        self.cfgsect = cfgsect
        self.name = cfgsect.removeprefix('soln-sliding-interface-')
        self.order = cfg.getint('solver', 'order')
        self.comm = sicomm

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
        self.ninters_lhs = len(self.lhs)
        self.ninters_rhs = len(self.rhs)
        self.ninterfpts_lhs = sum(elemap[etype].nfacefpts[fidx]
                                  for etype, eidx, fidx in self.lhs)
        self.ninterfpts_rhs = sum(elemap[etype].nfacefpts[fidx]
                                  for etype, eidx, fidx in self.rhs)

        # Get the local bounds on each face and the fpts locations
        self._set_original_fpts_and_bounds(self.lhs, self.rhs)

        # Gather the local bounds for each face and also the fpts locations
        self._gather_all_fpts_and_bounds()

        # Views, constant matrices and copies of face point data
        tags = {'align'}
        if self.ninters_lhs:
            self._scal_lhs = self._scal_view(self.lhs, 'get_scal_fpts_for_inter')
            self._pnorm_lhs = self._const_mat(self.lhs, 'get_pnorms_for_inter')
            mat_size_lhs = (self.nvars, self.ninterfpts_lhs)
            self._scal_lhs_copy = self._be.matrix(mat_size_lhs,
                                              tags=tags, extent=f'sliding_lhs_copy_{self.name}',
                                              initval=np.full(mat_size_lhs, 0.0))
            self._scal_lhs_interp = self._be.matrix(mat_size_lhs,
                                              tags=tags, extent=f'sliding_lhs_interp_{self.name}',
                                              initval=np.full(mat_size_lhs, 0.0))
        if self.ninters_rhs:
            self._scal_rhs = self._scal_view(self.rhs, 'get_scal_fpts_for_inter')
            self._pnorm_rhs = self._const_mat(self.rhs, 'get_pnorms_for_inter')
            mat_size_rhs = (self.nvars, self.ninterfpts_rhs)
            self._scal_rhs_copy = self._be.matrix(mat_size_rhs,
                                              tags=tags, extent=f'sliding_rhs_copy_{self.name}',
                                              initval=np.full(mat_size_rhs, 0.0))
            self._scal_rhs_interp = self._be.matrix(mat_size_rhs,
                                              tags=tags, extent=f'sliding_rhs_interp_{self.name}',
                                              initval=np.full(mat_size_rhs, 0.0))


        # Make the simulation time available inside kernels
        self._set_external('t', 'scalar fpdtype_t')

        self._entmin_lhs = None
        self._entmin_rhs = None
        if self._ef_enabled:
            if self.ninters_lhs:
                self._entmin_lhs = self._view(self.lhs, 'get_entmin_bc_fpts_for_inter')
            if self.ninters_rhs:
                self._entmin_rhs = self._view(self.rhs, 'get_entmin_bc_fpts_for_inter')

        # Matrices for interpolation
        max_ninterfpts = 200
        mat_size_fidx = (1, max_ninterfpts)
        zero_init = np.full(mat_size_fidx, 0, dtype=self._be.ixdtype)
        self._lhs_fidx = self._be.resizable_matrix(mat_size_fidx, tags=tags,
                                         initval=zero_init, dtype=self._be.ixdtype)
        self._rhs_fidx = self._be.resizable_matrix(mat_size_fidx, tags=tags,
                                         initval=zero_init, dtype=self._be.ixdtype)
        zero_init = np.full(mat_size_fidx, 0.0)
        self._lhs_rloc = self._be.resizable_matrix(mat_size_fidx, tags=tags,
                                         initval=zero_init)
        self._rhs_rloc = self._be.resizable_matrix(mat_size_fidx, tags=tags,
                                         initval=zero_init)
        mat_size_interp = (len(self.fpts), max_ninterfpts)
        zero_init = np.full(mat_size_interp, 0.0)
        self._lhs_interp_mats = self._be.resizable_matrix(mat_size_interp, tags=tags,
                                         initval=zero_init)
        self._rhs_interp_mats = self._be.resizable_matrix(mat_size_interp, tags=tags,
                                         initval=zero_init)
        mat_size_remote_results = (self.nvars, max_ninterfpts)
        zero_init = np.full(mat_size_remote_results, 0.0)
        self._interp_results_for_remote_lhs = self._be.resizable_matrix(mat_size_remote_results,
                                                tags=tags, initval=zero_init)
        self._interp_results_for_remote_rhs = self._be.resizable_matrix(mat_size_remote_results,
                                                tags=tags, initval=zero_init)
        
        # Kernels common across all solver
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.sicopy')
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.siinterp')
        self._be.pointwise.register('pyfr.solvers.baseadvec.kernels.sicalcmats')

        tplargs = dict(nvars=self.nvars)

        if self.ninters_lhs:
            self.kernels['copy_fpts_lhs'] = lambda: self._be.kernel(
                'sicopy', tplargs=tplargs, dims=[self.ninterfpts_lhs], 
                src=self._scal_lhs, dst=self._scal_lhs_copy
            )
        if self.ninters_rhs:
            self.kernels['copy_fpts_rhs'] = lambda: self._be.kernel(
                'sicopy', tplargs=tplargs, dims=[self.ninterfpts_rhs], 
                src=self._scal_rhs, dst=self._scal_rhs_copy
            )

        if self.ninters_rhs:
            self.kernels['interp_fpts_for_remote_lhs'] = lambda: self._be.kernel(
                'siinterp', tplargs=self._tplargs | dict(lhs=True, ninterfpts=self.ninterfpts_rhs), dims=[max_ninterfpts],
                src=self._scal_rhs_copy, fidx=self._rhs_fidx, mat=self._rhs_interp_mats,
                dst=self._interp_results_for_remote_lhs
            )
        if self.ninters_lhs:
            self.kernels['interp_fpts_for_remote_rhs'] = lambda: self._be.kernel(
                'siinterp', tplargs=self._tplargs | dict(lhs=False, ninterfpts=self.ninterfpts_lhs), dims=[max_ninterfpts],
                src=self._scal_lhs_copy, fidx=self._lhs_fidx, mat=self._lhs_interp_mats,
                dst=self._interp_results_for_remote_rhs
            )
        
        self._invvdm = self._be.const_matrix(self._face_polybasis.invvdm)
        if self.ninters_rhs:
            self.kernels['calc_mats_for_remote_lhs'] = lambda: self._be.kernel(
                'sicalcmats', tplargs=self._tplargs, dims=[max_ninterfpts],
                rloc=self._rhs_rloc, out=self._rhs_interp_mats, invvdm=self._invvdm
            )
        if self.ninters_lhs:
            self.kernels['calc_mats_for_remote_rhs'] = lambda: self._be.kernel(
                'sicalcmats', tplargs=self._tplargs, dims=[max_ninterfpts],
                rloc=self._lhs_rloc, out=self._lhs_interp_mats, invvdm=self._invvdm
            )
        
        self._prepare_time = 0.0
        self._prepare1_time = 0.0
        self._prepare2_time = 0.0
        self._prepare3_time = 0.0
        self._prepare4_time = 0.0
        self._prepare5_time = 0.0
        self._comm_time = 0.0
        self._tstart = time.time()
        self._t_counter = 0
    
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
        self._lhs_plocs = _get_inter_objs(lhs, 'get_plocs_for_inter', self.elemap) if self.ninters_lhs else []
        self._rhs_plocs = _get_inter_objs(rhs, 'get_plocs_for_inter', self.elemap) if self.ninters_rhs else []

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

    def _gather_all_fpts_and_bounds(self):
        # Get number of LHS and RHS elements on each rank with allgather
        self.global_lhs_counts = np.empty(self.comm.size, dtype=self._be.ixdtype)
        self.comm.Allgather(np.array([self.ninters_lhs], dtype=self._be.ixdtype), self.global_lhs_counts)
        self.global_rhs_counts = np.empty(self.comm.size, dtype=self._be.ixdtype)
        self.comm.Allgather(np.array([self.ninters_rhs], dtype=self._be.ixdtype), self.global_rhs_counts)

        # Do allgatherv for LHS face bounds
        lhs_sndbuf = np.array(self._lhs_face_bounds, dtype=self._be.fpdtype).flatten()
        lhs_rcvbuf = np.zeros((np.sum(self.global_lhs_counts)*2), dtype=self._be.fpdtype)
        lhs_disp = np.array([np.sum(self.global_lhs_counts[:i])*2 for i in range(0, len(self.global_lhs_counts))], dtype=self._be.ixdtype)
        self.comm.Allgatherv(lhs_sndbuf, [lhs_rcvbuf, (self.global_lhs_counts*2, lhs_disp)])
        self.global_lhs_bounds = []
        for disp, count in zip(lhs_disp, self.global_lhs_counts):
            self.global_lhs_bounds.append(np.reshape(lhs_rcvbuf[disp:disp+count*2], (-1, 2)))

        # Do allgatherv for RHS face bounds
        rhs_sndbuf = np.array(self._rhs_face_bounds, dtype=self._be.fpdtype).flatten()
        rhs_rcvbuf = np.zeros((np.sum(self.global_rhs_counts)*2), dtype=self._be.fpdtype)
        rhs_disp = np.array([np.sum(self.global_rhs_counts[:i])*2 for i in range(0, len(self.global_rhs_counts))], dtype=self._be.ixdtype)
        self.comm.Allgatherv(rhs_sndbuf, [rhs_rcvbuf, (self.global_rhs_counts*2, rhs_disp)])
        self.global_rhs_bounds = []
        for disp, count in zip(rhs_disp, self.global_rhs_counts):
            self.global_rhs_bounds.append(np.reshape(rhs_rcvbuf[disp:disp+count*2], (-1, 2)))
        
        # Do allgatherv for LHS face pts
        lhs_sndbuf = np.array(self._lhs_plocs, dtype=self._be.fpdtype).flatten()
        lhs_rcvbuf = np.zeros((np.sum(self.global_lhs_counts)*2*len(self.fpts)), dtype=self._be.fpdtype)
        lhs_disp = np.array([np.sum(self.global_lhs_counts[:i])*2*len(self.fpts) for i in range(0, len(self.global_lhs_counts))], dtype=self._be.ixdtype)
        self.comm.Allgatherv(lhs_sndbuf, [lhs_rcvbuf, (self.global_lhs_counts*2*len(self.fpts), lhs_disp)])
        self.global_lhs_plocs = []
        for disp, count in zip(lhs_disp, self.global_lhs_counts):
            self.global_lhs_plocs.append(np.reshape(lhs_rcvbuf[disp:disp+count*2*len(self.fpts)], (-1, 2)))

        # Do allgatherv for RHS pts
        rhs_sndbuf = np.array(self._rhs_plocs, dtype=self._be.fpdtype).flatten()
        rhs_rcvbuf = np.zeros((np.sum(self.global_rhs_counts)*2*len(self.fpts)), dtype=self._be.fpdtype)
        rhs_disp = np.array([np.sum(self.global_rhs_counts[:i])*2*len(self.fpts) for i in range(0, len(self.global_rhs_counts))], dtype=self._be.ixdtype)
        self.comm.Allgatherv(rhs_sndbuf, [rhs_rcvbuf, (self.global_rhs_counts*2*len(self.fpts), rhs_disp)])
        self.global_rhs_plocs = []
        for disp, count in zip(rhs_disp, self.global_rhs_counts):
            self.global_rhs_plocs.append(np.reshape(rhs_rcvbuf[disp:disp+count*2*len(self.fpts)], (-1, 2)))
        
        # Get global min and max for the face bounds (needed to mod transformed plocs)
        self.global_lhs_min_bound = min([np.min(bounds.flatten()) for bounds in self.global_lhs_bounds if len(bounds) > 0])
        self.global_lhs_max_bound = max([np.max(bounds.flatten()) for bounds in self.global_lhs_bounds if len(bounds) > 0])
        self.global_rhs_min_bound = min([np.min(bounds.flatten()) for bounds in self.global_rhs_bounds if len(bounds) > 0])
        self.global_rhs_max_bound = max([np.max(bounds.flatten()) for bounds in self.global_rhs_bounds if len(bounds) > 0])

    def _apply_transform_local(self, t):
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
        lhs_len = self.global_lhs_max_bound - self.global_lhs_min_bound
        rhs_len = self.global_rhs_max_bound - self.global_rhs_min_bound
        for _ploc in t_ploc_lhs:
            if _ploc[1] < self.global_rhs_min_bound:
                _ploc[1] += np.floor((self.global_rhs_max_bound - _ploc[1]) / rhs_len) * rhs_len
            if _ploc[1] > self.global_rhs_max_bound:
                _ploc[1] -= np.floor((_ploc[1] - self.global_rhs_min_bound) / rhs_len) * rhs_len
        
        for _ploc in t_ploc_rhs:
            if _ploc[1] < self.global_lhs_min_bound:
                _ploc[1] += np.floor((self.global_lhs_max_bound - _ploc[1]) / lhs_len) * lhs_len
            if _ploc[1] > self.global_lhs_max_bound:
                _ploc[1] -= np.floor((_ploc[1] - self.global_lhs_min_bound) / lhs_len) * lhs_len
        
        return t_ploc_lhs, t_ploc_rhs
    
    def _apply_transform_global(self, t):
        # Apply the transform to each side of the equation
        # Only transform the points, not the face bounds 
        # (so need to account for this in the transform)
        dist_u_l = 0.0 # (self.ul - self.ur) * t
        dist_v_l = (self.vl - self.vr) * t
        dist_u_r = 0.0 # (self.ur - self.ul) * t
        dist_v_r = (self.vr - self.vl) * t
        t_ploc_lhs = []
        for rank_ploc in self.global_lhs_plocs:
            t_ploc_lhs.append(np.array([[_u + dist_u_l, _v + dist_v_l] for _u, _v in rank_ploc]))
        t_ploc_rhs = []
        for rank_ploc in self.global_rhs_plocs:
            t_ploc_rhs.append(np.array([[_u + dist_u_r, _v + dist_v_r] for _u, _v in rank_ploc]))

        # Mod each ploc point to match the otherside's face bounds
        lhs_len = self.global_lhs_max_bound - self.global_lhs_min_bound
        rhs_len = self.global_rhs_max_bound - self.global_rhs_min_bound
        for rank_ploc in t_ploc_lhs:
            for _ploc in rank_ploc:
                if _ploc[1] < self.global_rhs_min_bound:
                    _ploc[1] += np.floor((self.global_rhs_max_bound - _ploc[1]) / rhs_len) * rhs_len
                if _ploc[1] > self.global_rhs_max_bound:
                    _ploc[1] -= np.floor((_ploc[1] - self.global_rhs_min_bound) / rhs_len) * rhs_len
        
        for rank_ploc in t_ploc_rhs:
            for _ploc in rank_ploc:
                if _ploc[1] < self.global_lhs_min_bound:
                    _ploc[1] += np.floor((self.global_lhs_max_bound - _ploc[1]) / lhs_len) * lhs_len
                if _ploc[1] > self.global_lhs_max_bound:
                    _ploc[1] -= np.floor((_ploc[1] - self.global_lhs_min_bound) / lhs_len) * lhs_len
        
        return t_ploc_lhs, t_ploc_rhs
    
    def _check_pt_in_line_face(self, pt, fbounds):
        return pt[1] + 1e-10 >= fbounds[0] and pt[1] - 1e-10 <= fbounds[1]
    
    # Brute force search for now
    def _get_rank_fidx_for_pts(self, pts_plocs, face_bounds):
        rank_fidx = []
        for _ploc in pts_plocs:
            _fidx = -1
            _rank = -1
            for r in range(0, len(face_bounds)):
                for i in range(0, len(face_bounds[r])):
                    if self._check_pt_in_face(_ploc, face_bounds[r][i]):
                        _fidx = i
                        _rank = r
                        break
                if _fidx != -1:
                    break
            if _fidx == -1:
                raise Exception(f'A sliding interface point ({_ploc[0]},{_ploc[1]}) is not within any faces')
            rank_fidx.append((_rank, _fidx))
        return rank_fidx

    # Brute force search for now
    def _get_fidx_rank_pidx(self, pts_plocs, face_bounds):
        interp_info = []
        for r in range(0, len(pts_plocs)):
            for pidx in range(0, len(pts_plocs[r])):
                for fidx in range(0, len(face_bounds)):
                    if self._check_pt_in_face(pts_plocs[r][pidx], face_bounds[fidx]):
                        interp_info.append((fidx, r, pidx))
                        break
        return interp_info

    # Assume y axis aligned line
    def _get_rloc(self, pt_ploc, face_bounds):
        return 2.0 * ((pt_ploc[1] - face_bounds[0]) / (face_bounds[1] - face_bounds[0])) - 1.0
    
    def _get_interp_mats_for_pts(self, pts_ploc, faces_bounds, interp_info):
        rlocs = []
        for fidx, rank, pidx in interp_info:
            face_bounds = faces_bounds[fidx]
            rloc = self._get_rloc(pts_ploc[rank][pidx], face_bounds)
            rlocs.append(rloc)
        return self._face_polybasis.nodal_basis_at(rlocs)
    
    def _get_rloc_for_pts(self, pts_ploc, faces_bounds, interp_info):
        rlocs = []
        for fidx, rank, pidx in interp_info:
            face_bounds = faces_bounds[fidx]
            rloc = self._get_rloc(pts_ploc[rank][pidx], face_bounds)
            rlocs.append(rloc)
        return np.array(rlocs)

    def prepare_interpolation(self, t, kerns):
        tstart = time.time()
        # Get the current plocs of each face point
        lhs_plocs, rhs_plocs = self._apply_transform_local(t)
        lhs_face_bounds, rhs_face_bounds = self.global_lhs_bounds, self.global_rhs_bounds
        self._prepare1_time += time.time() - tstart
        tstart1 = time.time()

        # Work out which rank and face contains each local face point
        self.lhs_pts_rhs_fidx = self._get_rank_fidx_for_pts(lhs_plocs, rhs_face_bounds)
        self.rhs_pts_lhs_fidx = self._get_rank_fidx_for_pts(rhs_plocs, lhs_face_bounds)
        self._prepare2_time += time.time() - tstart1
        tstart1 = time.time()

        # Work out which interpolations we'll need to perform before sending to other ranks
        global_lhs_plocs, global_rhs_plocs = self._apply_transform_global(t)
        self.lhs_interps_for_remote_rhs = self._get_fidx_rank_pidx(global_rhs_plocs, self._lhs_face_bounds)
        self.rhs_interps_for_remote_lhs = self._get_fidx_rank_pidx(global_lhs_plocs, self._rhs_face_bounds)
        self._prepare3_time += time.time() - tstart1
        tstart1 = time.time()

        # Calculate matrix to interpolate to face point
        # self.lhs_interp_matrices_for_remote_rhs = self._get_interp_mats_for_pts(global_rhs_plocs, self._lhs_face_bounds, self.lhs_interps_for_remote_rhs)
        # self.rhs_interp_matrices_for_remote_lhs = self._get_interp_mats_for_pts(global_lhs_plocs, self._rhs_face_bounds, self.rhs_interps_for_remote_lhs)
        self.rlocs_remote_rhs = self._get_rloc_for_pts(global_rhs_plocs, self._lhs_face_bounds, self.lhs_interps_for_remote_rhs)
        self.rlocs_remote_lhs = self._get_rloc_for_pts(global_lhs_plocs, self._rhs_face_bounds, self.rhs_interps_for_remote_lhs)
        self._prepare4_time += time.time() - tstart1
        tstart1 = time.time()

        # Update sizes of backend matrices
        self._lhs_fidx.resize((1, len(self.lhs_interps_for_remote_rhs)))
        self._rhs_fidx.resize((1, len(self.rhs_interps_for_remote_lhs)))
        self._lhs_rloc.resize((1, len(self.lhs_interps_for_remote_rhs)))
        self._rhs_rloc.resize((1, len(self.rhs_interps_for_remote_lhs)))
        self._lhs_interp_mats.resize((len(self.fpts), len(self.lhs_interps_for_remote_rhs)))
        self._rhs_interp_mats.resize((len(self.fpts), len(self.rhs_interps_for_remote_lhs)))
        self._interp_results_for_remote_lhs.resize((self.nvars, len(self.rhs_interps_for_remote_lhs)))
        self._interp_results_for_remote_rhs.resize((self.nvars, len(self.lhs_interps_for_remote_rhs)))

        # Set backend matrices
        lhs_fidx = np.array([[fidx for fidx, _, _ in self.lhs_interps_for_remote_rhs]])
        self._lhs_fidx.set(np.reshape(lhs_fidx, (-1, 1)).swapaxes(0,1))
        rhs_fidx = np.array([[fidx for fidx, _, _ in self.rhs_interps_for_remote_lhs]])
        self._rhs_fidx.set(np.reshape(rhs_fidx, (-1, 1)).swapaxes(0,1))
        # self._lhs_interp_mats.set(np.reshape(self.lhs_interp_matrices_for_remote_rhs, (-1, len(self.fpts))).swapaxes(0,1))
        # self._rhs_interp_mats.set(np.reshape(self.rhs_interp_matrices_for_remote_lhs, (-1, len(self.fpts))).swapaxes(0,1))
        self._lhs_rloc.set(np.reshape(self.rlocs_remote_rhs, (-1, 1)).swapaxes(0,1))
        self._rhs_rloc.set(np.reshape(self.rlocs_remote_lhs, (-1, 1)).swapaxes(0,1))

        # Update dims of interpolation kernels
        if self.ninters_rhs:
            kerns['interp_fpts_for_remote_lhs'].update_dims([len(self.rhs_interps_for_remote_lhs)])
            kerns['calc_mats_for_remote_lhs'].update_dims([len(self.rhs_interps_for_remote_lhs)])
        if self.ninters_lhs:
            kerns['interp_fpts_for_remote_rhs'].update_dims([len(self.lhs_interps_for_remote_rhs)])
            kerns['calc_mats_for_remote_rhs'].update_dims([len(self.lhs_interps_for_remote_rhs)])
        
        self._prepare5_time += time.time() - tstart1
        self._prepare_time += time.time() - tstart

    # Count number of interpolated points to expect from a rank
    def _count_recv_pts(self, rfinfo, rank):
        count = 0
        for _rank, _fidx in rfinfo:
            if rank == _rank:
                count += 1
        return count
    
    # Count number of interpolated points to send to this rank
    def _count_send_pts(self, iinfo, rank):
        count = 0
        for _fidx, _rank, _pidx in iinfo:
            if rank == _rank:
                count += 1
        return count
    
    # Pack a send buffer
    def _pack_send_buffer(self, iinfo, idata, buf, rank):
        bidx = 0
        for i in range(0, len(iinfo)):
            if iinfo[i][1] == rank:
                buf[bidx:bidx+self.nvars] = idata[:,i]
                bidx += self.nvars

    def _unpack_recv_buffers(self, rcv_bufs, local_buf, rfinfo):
        rank_counts = [0] * self.comm.size
        for i in range(0, len(rfinfo)):
            rank = rfinfo[i][0]
            if rank != self.comm.rank:
                rcv_idx = rank_counts[rank] * self.nvars
                local_buf[:,i] = rcv_bufs[rank][rcv_idx:rcv_idx+self.nvars]
                rank_counts[rank] += 1

    # TODO - do this via mako kernels, just doing it here for now to avoid issue with varying numbers of points across different time steps
    def interpolate(self):
        tstart = time.time()
        lhs2rhs_iinfo = self.lhs_interps_for_remote_rhs
        rhs2lhs_iinfo = self.rhs_interps_for_remote_lhs

        # Do the interpolation
        if self.ninters_lhs:
            idata_for_remote_rhs = self._interp_results_for_remote_rhs.get()
        if self.ninters_rhs:
            idata_for_remote_lhs = self._interp_results_for_remote_lhs.get()

        # Create send/recv buffers
        lhs_rcv_counts = []
        rhs_rcv_counts = []
        lhs_snd_counts = []
        rhs_snd_counts = []
        lhs_rcv_buffers = []
        rhs_rcv_buffers = []
        lhs_snd_buffers = []
        rhs_snd_buffers = []
        for rank in range(0, self.comm.size):
            # Skip local rank
            if rank == self.comm.rank:
                lhs_rcv_counts.append(0)
                rhs_rcv_counts.append(0)
                lhs_snd_counts.append(0)
                rhs_snd_counts.append(0)
                lhs_rcv_buffers.append(None)
                rhs_rcv_buffers.append(None)
                lhs_snd_buffers.append(None)
                rhs_snd_buffers.append(None)
                continue

            # Check how much we are expecting to receive from this rank
            lhs_recv = self._count_recv_pts(self.lhs_pts_rhs_fidx, rank)
            rhs_recv = self._count_recv_pts(self.rhs_pts_lhs_fidx, rank)
            lhs_rcv_counts.append(lhs_recv)
            rhs_rcv_counts.append(rhs_recv)

            # Create recv buffer
            lhs_rcv_buffers.append(np.zeros((self.nvars * lhs_recv), dtype=self._be.fpdtype))
            rhs_rcv_buffers.append(np.zeros((self.nvars * rhs_recv), dtype=self._be.fpdtype))

            # Check how much we are sending to this rank
            lhs_send = self._count_send_pts(rhs2lhs_iinfo, rank)
            rhs_send = self._count_send_pts(lhs2rhs_iinfo, rank)
            lhs_snd_counts.append(lhs_send)
            rhs_snd_counts.append(rhs_send)

            # Create send buffer
            lhs_snd_buffers.append(np.zeros((self.nvars * lhs_send), dtype=self._be.fpdtype))
            rhs_snd_buffers.append(np.zeros((self.nvars * rhs_send), dtype=self._be.fpdtype))


        # Buffer for final unpacked data
        local_lhs_interp = np.zeros((self.nvars, self.ninterfpts_lhs), dtype=self._be.fpdtype)
        local_rhs_interp = np.zeros((self.nvars, self.ninterfpts_rhs), dtype=self._be.fpdtype)

        # Send/Recv interpolated data
        mpi_requests = []
        for rank in range(0, self.comm.size):
            if rank == self.comm.rank:
                # Copy interpolated data that will stay locally
                for iidx in range(0, len(lhs2rhs_iinfo)):
                    if lhs2rhs_iinfo[iidx][1] == self.comm.rank:
                        local_rhs_interp[:,lhs2rhs_iinfo[iidx][2]] = idata_for_remote_rhs[:,iidx]
                for iidx in range(0, len(rhs2lhs_iinfo)):
                    if rhs2lhs_iinfo[iidx][1] == self.comm.rank:
                        local_lhs_interp[:,rhs2lhs_iinfo[iidx][2]] = idata_for_remote_lhs[:,iidx]
            else:
                # Non-blocking receive
                if lhs_rcv_counts[rank] > 0:
                    mpi_requests.append(self.comm.Irecv(lhs_rcv_buffers[rank], rank, self.LHS_MPI_TAG))
                if rhs_rcv_counts[rank] > 0:
                    mpi_requests.append(self.comm.Irecv(rhs_rcv_buffers[rank], rank, self.RHS_MPI_TAG))

                # Pack data to send to this rank and non-blocking send
                if lhs_snd_counts[rank] > 0:
                    self._pack_send_buffer(rhs2lhs_iinfo, idata_for_remote_lhs, lhs_snd_buffers[rank], rank)
                    mpi_requests.append(self.comm.Isend(lhs_snd_buffers[rank], rank, self.LHS_MPI_TAG))
                if rhs_snd_counts[rank] > 0:
                    self._pack_send_buffer(lhs2rhs_iinfo, idata_for_remote_rhs, rhs_snd_buffers[rank], rank)
                    mpi_requests.append(self.comm.Isend(rhs_snd_buffers[rank], rank, self.RHS_MPI_TAG))
        
        # Wait on non blocking comms
        for req in mpi_requests:
            req.Wait()
            req.free()
        
        # Unpack received data and update PyFR matrices
        if self.ninters_lhs:
            self._unpack_recv_buffers(lhs_rcv_buffers, local_lhs_interp, self.lhs_pts_rhs_fidx)
            self._scal_lhs_interp.set(local_lhs_interp)
        
        if self.ninters_rhs:
            self._unpack_recv_buffers(rhs_rcv_buffers, local_rhs_interp, self.rhs_pts_lhs_fidx)
            self._scal_rhs_interp.set(local_rhs_interp)
        
        self._comm_time += time.time() - tstart
        self._t_counter += 1

        if self._t_counter % 1000 == 0:
            print(f'{time.time() - self._tstart}, {self._prepare_time}, {self._comm_time} : {self._prepare1_time}, {self._prepare2_time}, {self._prepare3_time}, {self._prepare4_time}, {self._prepare5_time}')
