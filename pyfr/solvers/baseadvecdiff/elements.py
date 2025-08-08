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

        # Avoid self._comm_fpts being a row slice of a different buffer
        # Important as we are currently using a pointwise kernel to do
        # the batch mat mults
        bufs |= {'comm_fpts'}
        return bufs

    def set_backend(self, backend, nscalupts, nonce, linoff):
        super().set_backend(backend, nscalupts, nonce, linoff)

        kernel, kernels = self._be.kernel, self.kernels
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        slicem, slicedk = self._slice_mat, self._make_sliced_kernel

        # Register our pointwise kernels
        self._be.pointwise.register(f'{kprefix}.gradcoru')

        # Mesh regions
        regions = self._mesh_regions

        # First flux correction kernel (override which arrays to use if using physical flux)
        if self.phyf:
            # Allocate required vector scratch space
            alloc = lambda ex, n: backend.matrix(n, extent=nonce + ex, tags={'align'})
            valloc = lambda ex, n: alloc(ex, (self.ndims, n, self.nvars, self.neles))
            if 'vect_upts' in self._scratch_bufs:
                self._vect_upts_p = valloc('vect_upts_p', self.nupts)
            if 'vect_qpts' in self._scratch_bufs:
                self._vect_qpts_p = valloc('vect_qpts_p', self.nqpts)

            if 'flux' in self.antialias and self.basis.order > 0:
                kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                    'batchmm', dims=[self.neles], 
                    tplargs={'na': self.nupts, 'nb': self.nqpts*self.ndims, 'nvars': self.nvars, 'beta': 0.0},
                    A=self.opmat('(M111 - M3*M222)*M9'), u=self._vect_qpts_p,
                    v=self.scal_upts[fout]
                )
            elif self.basis.order > 0:
                kernels['tdivtpcorf'] = lambda fout: self._be.kernel(
                    'batchmm', dims=[self.neles],
                    tplargs={'na': self.nupts, 'nb': self.nupts*self.ndims, 'nvars': self.nvars, 'beta': 0.0},
                    A=self.opmat('M111 - M3*M222'), u=self._vect_upts_p, v=self.scal_upts[fout]
                )

        if abs(self.cfg.getfloat('solver-interfaces', 'ldg-beta')) == 0.5:
            kernels['copy_fpts'] = lambda: kernel(
                'copy', self._comm_fpts, self._scal_fpts
            )

        if self.basis.order > 0:
            if self.phyf:
                kernels['tgradpcoru_upts'] = lambda uin: kernel(
                    'batchmm', dims=[self.neles],
                    tplargs={'na': self.nupts*self.ndims, 'nb': self.nupts, 'nvars': self.nvars, 'beta': 0.0},
                    A=self.opmat('M444 - M6*M0'), u=self.scal_upts[uin],
                    v=self._grad_upts
                )
            else:
                kernels['tgradpcoru_upts'] = lambda uin: kernel(
                    'mul', self.opmat('M4 - M6*M0'), self.scal_upts[uin],
                    out=self._grad_upts
                )

        kernels['tgradcoru_upts'] = lambda: kernel(
            'mul', self.opmat('M6'), self._comm_fpts,
            out=self._grad_upts, beta=float(self.basis.order > 0)
        )

        # Template arguments for the physical gradient kernel
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'jac_exprs': self.basis.jac_exprs
        }

        gradcoru_u = []
        if 'curved' in regions:
            gradcoru_u.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'curved', 'phyf': self.phyf},
                dims=[self.nupts, regions['curved']],
                gradu=slicem(self._grad_upts, 'curved'),
                smats=self.curved_smat_at('upts'),
                rcpdjac=self.rcpdjac_at('upts', 'curved')
            ))
        if 'linear' in regions:
            gradcoru_u.append(lambda: kernel(
                'gradcoru', tplargs=tplargs | {'ktype': 'linear', 'phyf': self.phyf},
                dims=[self.nupts, regions['linear']],
                gradu=slicem(self._grad_upts, 'linear'),
                upts=self.upts, verts=self.ploc_at('linspts', 'linear')
            ))

        kernels['gradcoru_u'] = lambda: slicedk(k() for k in gradcoru_u)

        if not self.grad_fusion or self.basis.order == 0:
            kernels['gradcoru_upts'] = kernels['gradcoru_u']

        def gradcoru_fpts():
            nupts, nfpts = self.nupts, self.nfpts
            vupts, vfpts = self._grad_upts, self._vect_fpts

            # Exploit the block-diagonal form of the operator
            muls = [kernel('mul', self.opmat('M0'),
                           vupts.slice(i*nupts, (i + 1)*nupts),
                           vfpts.slice(i*nfpts, (i + 1)*nfpts))
                    for i in range(self.ndims)]

            return self._be.unordered_meta_kernel(muls)

        if not self.basis.fpts_in_upts:
            kernels['gradcoru_fpts'] = gradcoru_fpts

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
