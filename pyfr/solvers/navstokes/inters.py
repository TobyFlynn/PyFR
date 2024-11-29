import numpy as np

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import (FluidIntIntersMixin,
                                       FluidMPIIntersMixin)

import math
from collections import defaultdict, deque
from pyfr.quadrules import get_quadrule
from pyfr.mpiutil import mpi
from pyfr.inifile import NoOptionError
from pyfr.plugins.base import init_csv

class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min)


class NavierStokesIntInters(TplargsMixin,
                            FluidIntIntersMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            FluidMPIIntersMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artviscl=self._artvisc_lhs, **self._external_vals
        )

        if self._ef_enabled:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.bccent'
            )
            self._tplargs['e_func'] = self.cfg.get('solver-entropy-filter',
                                                   'e-func', 'numerical')

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, entmin_lhs=self._entmin_lhs,
                nl=self._pnorm_lhs, ul=self._scal_lhs, **self._external_vals
            )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(['p'], lhs)

# Boundary class that sets a mass flow across a boundary by varying
# the static pressure
class NavierStokesCharRiemInvMassFlowBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.gamma = self.cfg.getfloat('constants', 'gamma')
        # Name of inflow BC
        self.inflow_bc_name = self.cfg.get(cfgsect, 'inflow-name')
        # CpTt and Pt from inflow BC
        self.cpTt = self.cfg.getfloat('soln-bcs-' + self.inflow_bc_name, 'cpTt')
        self.pt = self.cfg.getfloat('soln-bcs-' + self.inflow_bc_name, 'pt')
        # Target Mach number
        self.m = self.cfg.getfloat(cfgsect, 'm')
        self.outlet_bc_name = cfgsect[9:]
        # Mass flow history
        self.mf_hist_len = 100
        self.mf_hist = deque(maxlen=self.mf_hist_len)
        # Values for ODE
        self.alpha = self.cfg.getfloat(cfgsect, 'alpha', 1.0)
        self.eta = self.cfg.getfloat(cfgsect, 'eta', 2.0)
        self.epsilon = self.cfg.getfloat(cfgsect, 'epsilon', 30)
        # MPI comm that only includes ranks that have this boundary
        self.bccomm = bccomm

        self.p = -1.0
        self.dpdt = 0.0
        self.tprev = -1.0

        self.elemap_copy = elemap

        self.target_mass_flow_rate = None

        # Constants for Mako kernel
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )

        self._set_external('var_p', 'scalar fpdtype_t')

        if self.bccomm.rank == 0:
            self.outf = init_csv(self.cfg, cfgsect, 't,mf,p,pbc')
    
    # Setup integrating over boundary
    def init_surface_integration(self, system, bc_name):
        # Underlying elements class
        elementscls = system.elementscls
        # Boundary to integrate over
        bc = f'bcon_{bc_name}_p{system.rallocs.prank}'
        # Get the mesh and elements
        mesh, elemap = system.mesh, self.elemap_copy
        # Interpolation matrices and quadrature weights
        _m0 = m0 = {}
        _qwts = qwts = defaultdict(list)
        _eidxs = None
        _norms = None
        _rfpts = None
        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices, associated face normals and relative flux
            # points position with respect to the moments origin
            eidxs = defaultdict(list)
            norms = defaultdict(list)
            rfpts = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].tolist():
                eles = elemap[etype]
                itype, proj, norm = eles.basis.faces[fidx]

                ppts, pwts = self._surf_quad(itype, proj, flags='s')
                nppts = len(ppts)

                # Get phyical normals
                pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(pnorm)

                if (etype, fidx) not in m0:
                    m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                    qwts[etype, fidx] = pwts

            _eidxs = {k: np.array(v) for k, v in eidxs.items()}
            _norms = {k: np.array(v) for k, v in norms.items()}
            _rfpts = {k: np.array(v) for k, v in rfpts.items()}
        return elementscls, _m0, _qwts, _eidxs, _norms, _rfpts
    
    def set_target_mass_flow_rate(self, system, soln):
        # Get target mass flow rate which should set the target Mach number at the inflow
        # Currently assumes that inflow area is the same as the outflow area
        self.inflow_area = self.calculate_area(system, soln)
        self.target_mass_flow_rate = self.inflow_area * (self.gamma / math.sqrt(self.gamma - 1.0)) \
                                    * (self.pt / math.sqrt(self.cpTt)) * self.m \
                                    * math.pow(1.0 + ((self.gamma - 1.0) / 2.0) * (self.m**2), (-self.gamma -1.0) / (2.0 * (self.gamma - 1.0)))

    def calculate_area(self, system, soln):
        solns = dict(zip(system.ele_types, system.ele_scal_upts(soln)))
        ndims, nvars = self.ndims, self.nvars
        sol_sizes = {}
        # Get the sizes for the area calculation
        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Compute the pressure
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[-1]
            sol_sizes[etype, fidx] = p.shape
        fm = np.zeros((ndims))
        # Get the sizes for the area calculation
        for etype, fidx in self._m0:
            # Array with ones so we can get area
            area_ones = np.ones(sol_sizes[etype, fidx])

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature
            fm[:ndims] += np.einsum('i...,ij,jik', qwts, area_ones, norms)
        self.bccomm.Allreduce(mpi.IN_PLACE, fm, op=mpi.SUM)
        return abs(sum(fm))
    
    def calculate_mass_flow(self, solns):
        ndims, nvars = self.ndims, self.nvars
        fm = np.zeros((ndims))
        # Get the sizes for the area calculation
        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature for each dimension
            # RhoU = ufpts[1], RhoV = ufpts[2], RhoW = ufpts[2]
            for i in range(0, ndims):
                rhoVel = ufpts[1 + i]
                fm[i] += np.einsum('i...,ij,ji', qwts, rhoVel, norms[:,:,i])
        self.bccomm.Allreduce(mpi.IN_PLACE, fm, op=mpi.SUM)
        return sum(fm)
    
    def calculate_p(self, solns):
        ndims, nvars = self.ndims, self.nvars
        fm = np.zeros((ndims))
        # Get the sizes for the area calculation
        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[-1]

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature for each dimension
            fm[:ndims] += np.einsum('i...,ij,jik', qwts, p, norms)
        self.bccomm.Allreduce(mpi.IN_PLACE, fm, op=mpi.SUM)
        return sum(fm) / self.inflow_area
    
    def update_mf(self, solns):
        mf = self.calculate_mass_flow(solns)
        self.mf_hist.append(mf)
    
    def avg_mf(self):
        return np.mean(self.mf_hist) if self.mf_hist else 0.0
    
    def p_ode(self, y, avg_mf, target_mf):
        p, dpdt = y
        alpha = self.alpha
        eta = self.eta
        epsilon = self.epsilon
        d2pdt2 = (eta * (avg_mf - target_mf) - epsilon * dpdt) / alpha
        return [dpdt, d2pdt2]
    
    def p_rk4_step(self, dt, y, avg_mf, target_mf):
        k1 = self.p_ode(y, avg_mf, target_mf)
        k2 = self.p_ode([x0 + x1 * dt * 0.5 for x0, x1 in zip(y, k1)], avg_mf, target_mf)
        k3 = self.p_ode([x0 + x1 * dt * 0.5 for x0, x1 in zip(y, k2)], avg_mf, target_mf)
        k4 = self.p_ode([x0 + x1 * dt for x0, x1 in zip(y, k3)], avg_mf, target_mf)
        return [x0 + (dt / 6.0) * (x1 + 2.0 * x2 + 2.0 * x3 + x4) for x0, x1, x2, x3, x4 in zip(y, k1, k2, k3, k4)]


    def update_p(self, dt):
        avg_mf = self.avg_mf()
        self.p, self.dpdt = self.p_rk4_step(dt, [self.p, self.dpdt], avg_mf, self.target_mass_flow_rate)
    
    def prepare(self, t, system, soln):
        # Check if first prepare call
        if self.target_mass_flow_rate is None:
            self.ndims = system.ndims
            self.nvars = system.nvars
            self.elementscls, self._m0, self._qwts, self._eidxs, self._norms, self._rfpts = self.init_surface_integration(system, self.outlet_bc_name)
            del self.elemap_copy
            self.set_target_mass_flow_rate(system, soln)

        solns = dict(zip(system.ele_types, system.ele_scal_upts(soln)))
        self.update_mf(solns)
        if self.tprev < 0.0:
            self.tprev = t
            self.update_mf(solns)
            self.p = self.calculate_p(solns)
            system.update_kernel_extern('var_p', self.p)
            return

        self.update_p(t - self.tprev)
        system.update_kernel_extern('var_p', self.p)
        self.tprev = t

        # Output mass flow and pressure at outflow
        mass_flow = self.calculate_mass_flow(solns)
        p_force = self.calculate_p(solns)
        # Save values to CSV file
        if self.bccomm.rank == 0:
            print(f'{t},{mass_flow},{p_force},{self.p}', file=self.outf)
            self.outf.flush()
    
    # Copied from plugins/base.py:SurfaceMixin
    def _surf_quad(self, itype, proj, flags=''):
        # Obtain quadrature info
        rname = self.cfg.get(f'solver-interfaces-{itype}', 'flux-pts')

        # Quadrature rule (default to that of the solution points)
        qrule = self.cfg.get(self.cfgsect, f'quad-pts-{itype}', rname)
        try:
            qdeg = self.cfg.getint(self.cfgsect, f'quad-deg-{itype}')
        except NoOptionError:
            qdeg = self.cfg.getint(self.cfgsect, 'quad-deg')

        # Get the quadrature rule
        q = get_quadrule(itype, qrule, qdeg=qdeg, flags=flags)

        # Project its points onto the provided surface
        pts = np.atleast_2d(q.pts.T)
        return np.vstack(np.broadcast_arrays(*proj(*pts))).T, q.wts