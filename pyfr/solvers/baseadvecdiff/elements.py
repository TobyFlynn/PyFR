from pyfr.polys import get_polybasis
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseAdvectionDiffusionElements(BaseAdvectionElements):
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'vect_upts'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}
        elif self.grad_fusion:
            bufs |= {'grad_upts'}

        if self.basis.fpts_in_upts:
            bufs |= {'comm_fpts'}
            bufs -= {'vect_fpts'}

        return bufs

    def set_backend(self, backend, nscalupts, nonce, linoff, coff):
        super().set_backend(backend, nscalupts, nonce, linoff, coff)

        kernel, kernels = self._be.kernel, self.kernels
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        slicem, slicedk = self._slice_mat, self._make_sliced_kernel

        # Register our pointwise kernels
        self._be.pointwise.register(f'{kprefix}.gradcoru')

        # Mesh regions
        regions = self._mesh_regions

        if abs(self.cfg.getfloat('solver-interfaces', 'ldg-beta')) == 0.5:
            kernels['copy_fpts'] = lambda: kernel(
                'copy', self._comm_fpts, self._scal_fpts
            )

        if self.basis.order > 0:
            kernels['tgradpcoru_upts'] = lambda uin: kernel(
                'mul', self.opmat('M4 - M6*M0'), self.scal_upts[uin],
                out=self._grad_upts
            )
            for r in ['mpi', 'core']:
                if r in regions:
                    kernels[f'tgradpcoru_upts_{r}'] = lambda uin, r=r: kernel(
                        'mul', self.opmat('M4 - M6*M0'), slicem(self.scal_upts[uin], r),
                        out=slicem(self._grad_upts, r)
                    )

        kernels['tgradcoru_upts'] = lambda: kernel(
            'mul', self.opmat('M6'), self._comm_fpts,
            out=self._grad_upts, beta=float(self.basis.order > 0)
        )
        for r in ['mpi', 'core']:
            if r in regions:
                kernels[f'tgradcoru_upts_{r}'] = lambda r=r: kernel(
                    'mul', self.opmat('M6'), slicem(self._comm_fpts, r),
                    out=slicem(self._grad_upts, r), beta=float(self.basis.order > 0)
                )

        # Template arguments for the physical gradient kernel
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'jac_exprs': self.basis.jac_exprs
        }

        gradcoru_u = []
        gradcoru_u_core = []
        if 'curved' in regions:
            gradcoru_u.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, regions['curved']],
                gradu=slicem(self._grad_upts, 'curved'),
                smats=self.curved_smat_at('upts'),
                rcpdjac=self.rcpdjac_at('upts', 'curved')
            ))
        if 'linear' in regions:
            lin_kern = lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nupts, regions['linear']],
                gradu=slicem(self._grad_upts, 'linear'),
                upts=self.upts, verts=self.ploc_at('linspts', 'linear')
            )
            gradcoru_u.append(lin_kern)
            gradcoru_u_core.append(lin_kern)
        if 'corecurved' in regions:
            gradcoru_u_core.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, regions['corecurved']],
                gradu=slicem(self._grad_upts, 'corecurved'),
                smats=slicem(self.curved_smat_at('upts'), 'corecurved'),
                rcpdjac=self.rcpdjac_at('upts', 'corecurved')
            ))

        kernels['gradcoru_u'] = lambda: slicedk(k() for k in gradcoru_u)
        kernels['gradcoru_u_core'] = lambda: slicedk(k() for k in gradcoru_u_core)

        if 'mpi' in regions:
            kernels['gradcoru_u_mpi'] = lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, regions['mpi']],
                gradu=slicem(self._grad_upts, 'mpi'),
                smats=slicem(self.curved_smat_at('upts'), 'mpi'),
                rcpdjac=self.rcpdjac_at('upts', 'mpi')
            )

        if not self.grad_fusion or self.basis.order == 0:
            kernels['gradcoru_upts'] = kernels['gradcoru_u']
            kernels['gradcoru_upts_mpi'] = kernels['gradcoru_u_mpi']
            kernels['gradcoru_upts_core'] = kernels['gradcoru_u_core']

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._grad_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           vupts.slice(i*nupts, (i + 1)*nupts),
                           vfpts.slice(i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return self._be.unordered_meta_kernel(muls)

        def gradcoru_fpts_mpi():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._grad_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           self._slice_mat(vupts, 'mpi', i*nupts, (i + 1)*nupts),
                           self._slice_mat(vfpts, 'mpi', i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return self._be.unordered_meta_kernel(muls)
        
        def gradcoru_fpts_core():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._grad_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           self._slice_mat(vupts, 'core', i*nupts, (i + 1)*nupts),
                           self._slice_mat(vfpts, 'core', i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return self._be.unordered_meta_kernel(muls)

        if not self.basis.fpts_in_upts:
            kernels['gradcoru_fpts'] = gradcoru_fpts
            if 'mpi' in regions:
                kernels['gradcoru_fpts_mpi'] = gradcoru_fpts_mpi
            kernels['gradcoru_fpts_core'] = gradcoru_fpts_core

        if 'flux' in self.antialias and self.basis.order > 0:
            def gradcoru_qpts():
                nupts, nqpts = self.nupts, self.nqpts
                vupts, vqpts = self._vect_upts, self._vect_qpts

                # Exploit the block-diagonal form of the operator
                muls = [kernel('mul', self.opmat('M7'),
                               vupts.slice(i*nupts, (i + 1)*nupts),
                               vqpts.slice(i*nqpts, (i + 1)*nqpts))
                        for i in range(self.ndims)]

                return self._be.unordered_meta_kernel(muls)

            kernels['gradcoru_qpts'] = gradcoru_qpts

        # Shock capturing
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'artificial-viscosity':
            tags = {'align'}

            # Register the kernels
            self._be.pointwise.register(f'{kprefix}.shocksensor')

            # Obtain the scalar variable to be used for shock sensing
            shockvar = self.convars.index(self.shockvar)

            # Obtain the name, degrees, and order of our solution basis
            ubname = self.basis.ubasis.name
            ubdegs = self.basis.ubasis.degrees
            uborder = self.basis.ubasis.order

            # Obtain the degrees of a basis whose order is one lower
            lubdegs = get_polybasis(ubname, max(0, uborder - 1)).degrees

            # Compute the intersection
            ind_modes = [d not in lubdegs for d in ubdegs]

            # Template arguments
            tplargs_artvisc = dict(
                nvars=self.nvars, nupts=self.nupts, svar=shockvar,
                c=self.cfg.items_as('solver-artificial-viscosity', float),
                order=self.basis.order, ind_modes=ind_modes,
                invvdm=self.basis.ubasis.invvdm.T
            )

            # Allocate space for the artificial viscosity vector
            self.artvisc = self._be.matrix((1, self.neles),
                                           extent=nonce + 'artvisc', tags=tags)

            # Apply the sensor to estimate the required artificial viscosity
            kernels['shocksensor'] = lambda uin: kernel(
                'shocksensor', tplargs=tplargs_artvisc, dims=[self.neles],
                u=self.scal_upts[uin], artvisc=self.artvisc
            )
        elif shock_capturing in {'entropy-filter', 'none'}:
            self.artvisc = None
        else:
            raise ValueError('Invalid shock capturing scheme')

    def get_artvisc_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]
        return (self.artvisc.mid,)*nfp, (0,)*nfp, (eidx,)*nfp
