import numpy as np
import time

from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters,
                                    BaseAdvectionSlidingInters)


class BaseAdvectionDiffusionIntInters(BaseAdvectionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Generate the additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inter')
        self._vect_rhs = self._vect_view(rhs, 'get_vect_fpts_for_inter')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inter')
        self._comm_rhs = self._scal_view(rhs, 'get_comm_fpts_for_inter')

        # Generate the additional view matrices for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._view(lhs, 'get_artvisc_fpts_for_inter')
            self._artvisc_rhs = self._view(rhs, 'get_artvisc_fpts_for_inter')
        else:
            self._artvisc_lhs = self._artvisc_rhs = None

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

    def _gen_perm(self, lhs, rhs):
        # In the special case of β = -0.5 it is better to sort by the
        # RHS interface; otherwise we simply opt for the LHS
        beta = self.cfg.getfloat('solver-interfaces', 'ldg-beta')
        side = lhs if beta != -0.5 else rhs

        # Compute the relevant permutation
        self._perm = self._get_perm_for_view(side, 'get_scal_fpts_for_inter')


class BaseAdvectionDiffusionMPIInters(BaseAdvectionMPIInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, rhsrank, elemap, cfg)

        comm, rank, root = get_comm_rank_root()

        lhsprank = rank
        rhsprank = rhsrank

        # Generate second set of view matrices
        self._vect_lhs = self._vect_xchg_view(lhs, 'get_vect_fpts_for_inter')
        self._vect_rhs = be.xchg_matrix_for_view(self._vect_lhs)
        self._comm_lhs = self._scal_xchg_view(lhs, 'get_comm_fpts_for_inter')
        self._comm_rhs = be.xchg_matrix_for_view(self._comm_lhs)

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # We require cflux(l,r,n_l) = -cflux(r,l,n_r) and
        # conu(l,r) = conu(r,l) and where l and r are left and right
        # solutions at an interface and n_[l,r] are physical normals.
        # The simplest way to enforce this at an MPI interface is for
        # one side to take β = -β for the cflux and conu kernels. We
        # pick this side (arbitrarily) by comparing the physical ranks
        # of the two partitions.
        if (lhsprank + rhsprank) % 2:
            self.c['ldg-beta'] *= 1.0 if lhsprank > rhsprank else -1.0
        else:
            self.c['ldg-beta'] *= 1.0 if rhsprank > lhsprank else -1.0

        # Allocate a tag
        vect_fpts_tag = next(self._mpi_tag_counter)

        # If we need to send our gradients to the RHS
        if self.c['ldg-beta'] != -0.5:
            self.kernels['vect_fpts_pack'] = lambda: be.kernel(
                'pack', self._vect_lhs
            )
            self.mpireqs['vect_fpts_send'] = lambda: self._vect_lhs.sendreq(
                self._rhsrank, vect_fpts_tag
            )

        # If we need to recv gradients from the RHS
        if self.c['ldg-beta'] != 0.5:
            self.mpireqs['vect_fpts_recv'] = lambda: self._vect_rhs.recvreq(
                self._rhsrank, vect_fpts_tag
            )
            self.kernels['vect_fpts_unpack'] = lambda: be.kernel(
                'unpack', self._vect_rhs
            )

        # Generate the additional kernels/views for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._xchg_view(lhs,
                                                'get_artvisc_fpts_for_inter')
            self._artvisc_rhs = be.xchg_matrix_for_view(self._artvisc_lhs)

            # Allocate a tag
            artvisc_fpts_tag = next(self._mpi_tag_counter)

            # If we need to send our artificial viscosity to the RHS
            if self.c['ldg-beta'] != -0.5:
                av_lhs = self._artvisc_lhs
                self.kernels['artvisc_fpts_pack'] = lambda: be.kernel(
                    'pack', av_lhs
                )
                self.mpireqs['artvisc_fpts_send'] = lambda: av_lhs.sendreq(
                    self._rhsrank, artvisc_fpts_tag
                )

            # If we need to recv artificial viscosity from the RHS
            if self.c['ldg-beta'] != 0.5:
                av_rhs = self._artvisc_rhs
                self.mpireqs['artvisc_fpts_recv'] = lambda: av_rhs.recvreq(
                    self._rhsrank, artvisc_fpts_tag
                )
                self.kernels['artvisc_fpts_unpack'] = lambda: be.kernel(
                    'unpack', av_rhs
                )
        else:
            self._artvisc_lhs = self._artvisc_rhs = None


class BaseAdvectionDiffusionBCInters(BaseAdvectionBCInters):
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Additional view matrices
        self._vect_lhs = self._vect_view(lhs, 'get_vect_fpts_for_inter')
        self._comm_lhs = self._scal_view(lhs, 'get_comm_fpts_for_inter')

        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # Generate the additional view matrices for artificial viscosity
        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            self._artvisc_lhs = self._view(lhs, 'get_artvisc_fpts_for_inter')
        else:
            self._artvisc_lhs = None


class BaseAdvectionDiffusionSlidingInters(BaseAdvectionSlidingInters):
    LHS_GRAD_MPI_TAG = 3458
    RHS_GRAD_MPI_TAG = 3459

    def __init__(self, be, lhs, elemap, cfgsect, cfg, sicomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, sicomm)

        if cfg.get('solver', 'shock-capturing') == 'artificial-viscosity':
            raise Exception(f'artificial viscosity and sliding interfaces has not been implemented')
        
        # Additional kernel constants
        self.c |= cfg.items_as('solver-interfaces', float)

        # Copies of face point data
        tags = {'align'}
        if self.ninters_lhs:
            self._vect_lhs = self._vect_view(self.lhs, 'get_vect_fpts_for_inter')
            self._comm_lhs = self._scal_view(self.lhs, 'get_comm_fpts_for_inter')
            mat_size_copy = (self.ndims * self.nvars, self.ninterfpts_lhs)
            self._vect_lhs_copy = self._be.matrix(mat_size_copy,
                                              tags=tags, extent=f'sliding_lhs_copy_{self.name}',
                                              initval=np.full(mat_size_copy, 0.0))
            mat_size = (self.ndims, self.nvars, self.ninterfpts_lhs)
            self._vect_lhs_interp = self._be.matrix(mat_size,
                                                tags=tags, extent=f'sliding_lhs_interp_{self.name}',
                                                initval=np.full(mat_size, 0.0))
        if self.ninters_rhs:
            self._vect_rhs = self._vect_view(self.rhs, 'get_vect_fpts_for_inter')
            self._comm_rhs = self._scal_view(self.rhs, 'get_comm_fpts_for_inter')
            mat_size_copy = (self.ndims * self.nvars, self.ninterfpts_rhs)
            self._vect_rhs_copy = self._be.matrix(mat_size_copy,
                                              tags=tags, extent=f'sliding_rhs_copy_{self.name}',
                                              initval=np.full(mat_size_copy, 0.0))
            mat_size = (self.ndims, self.nvars, self.ninterfpts_rhs)
            self._vect_rhs_interp = self._be.matrix(mat_size,
                                                tags=tags, extent=f'sliding_rhs_interp_{self.name}',
                                                initval=np.full(mat_size, 0.0))
        
        mat_size_remote_results = (self.ndims, self.nvars, self.max_ninterfpts)
        zero_init = np.full(mat_size_remote_results, 0.0)
        self._interp_results_grad_for_remote_lhs = self._be.resizable_matrix(mat_size_remote_results,
                                                tags=tags, initval=zero_init)
        self._interp_results_grad_for_remote_rhs = self._be.resizable_matrix(mat_size_remote_results,
                                                tags=tags, initval=zero_init)
        
        self._be.pointwise.register('pyfr.solvers.baseadvecdiff.kernels.sicopygrad')
        self._be.pointwise.register('pyfr.solvers.baseadvecdiff.kernels.siinterpgrad')

        tplargs = dict(nvars=self.nvars, ndims=self.ndims)

        if self.ninters_lhs:
            self.kernels['copy_fpts_grad_lhs'] = lambda: self._be.kernel(
                'sicopygrad', tplargs=tplargs, dims=[self.ninterfpts_lhs], 
                src=self._vect_lhs, dst=self._vect_lhs_copy
            )

        if self.ninters_rhs:
            self.kernels['copy_fpts_grad_rhs'] = lambda: self._be.kernel(
                'sicopygrad', tplargs=tplargs, dims=[self.ninterfpts_rhs], 
                src=self._vect_rhs, dst=self._vect_rhs_copy
            )

        if self.ninters_rhs:
            self.kernels['interp_fpts_grad_for_remote_lhs'] = lambda: self._be.kernel(
                'siinterpgrad', tplargs=self._tplargs | dict(ninterfpts=self.ninterfpts_rhs), dims=[self.max_ninterfpts],
                src=self._vect_rhs_copy, fidx=self._rhs_fidx, mat=self._rhs_interp_mats,
                dst=self._interp_results_grad_for_remote_lhs
            )
        
        if self.ninters_lhs:
            self.kernels['interp_fpts_grad_for_remote_rhs'] = lambda: self._be.kernel(
                'siinterpgrad', tplargs=self._tplargs | dict(ninterfpts=self.ninterfpts_lhs), dims=[self.max_ninterfpts],
                src=self._vect_lhs_copy, fidx=self._lhs_fidx, mat=self._lhs_interp_mats,
                dst=self._interp_results_grad_for_remote_rhs
            )
        
        self._grad_comm_time = 0.0
    
    def prepare_interpolation(self, t, kerns):
        super().prepare_interpolation(t, kerns)

        # Update sizes of backend matrices
        self._interp_results_grad_for_remote_lhs.resize((self.ndims, self.nvars, max(len(self.rhs_interps_for_remote_lhs),1)))
        self._interp_results_grad_for_remote_rhs.resize((self.ndims, self.nvars, max(len(self.lhs_interps_for_remote_rhs),1)))

        # Update dims of interpolation kernels
        if self.ninters_rhs:
            kerns['interp_fpts_grad_for_remote_lhs'].update_dims([len(self.rhs_interps_for_remote_lhs)])
        if self.ninters_lhs:
            kerns['interp_fpts_grad_for_remote_rhs'].update_dims([len(self.lhs_interps_for_remote_rhs)])
    
    def _pack_send_grad_buffer(self, iinfo, idata, buf, rank):
        bidx = 0
        for i in range(0, len(iinfo)):
            if iinfo[i][1] == rank:
                buf[bidx:bidx+self.nvars*self.ndims] = np.reshape(idata[:,:,i], (-1))
                bidx += self.nvars*self.ndims

    def _unpack_recv_grad_buffers(self, rcv_bufs, local_buf, rfinfo):
        rank_counts = [0] * self.comm.size
        for i in range(0, len(rfinfo)):
            rank = rfinfo[i][0]
            if rank != self.comm.rank:
                rcv_idx = rank_counts[rank] * self.nvars * self.ndims
                local_buf[:,:,i] = np.reshape(rcv_bufs[rank][rcv_idx:rcv_idx+self.nvars*self.ndims],(self.ndims, self.nvars))
                rank_counts[rank] += 1

    def interpolate_grad(self):
        tstart = time.time()
        lhs2rhs_iinfo = self.lhs_interps_for_remote_rhs
        rhs2lhs_iinfo = self.rhs_interps_for_remote_lhs

        # Do the interpolation
        if self.ninters_lhs:
            idata_for_remote_rhs = self._interp_results_grad_for_remote_rhs.get()
        if self.ninters_rhs:
            idata_for_remote_lhs = self._interp_results_grad_for_remote_lhs.get()

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
            lhs_rcv_buffers.append(np.zeros((self.ndims * self.nvars * lhs_recv), dtype=self._be.fpdtype))
            rhs_rcv_buffers.append(np.zeros((self.ndims * self.nvars * rhs_recv), dtype=self._be.fpdtype))

            # Check how much we are sending to this rank
            lhs_send = self._count_send_pts(rhs2lhs_iinfo, rank)
            rhs_send = self._count_send_pts(lhs2rhs_iinfo, rank)
            lhs_snd_counts.append(lhs_send)
            rhs_snd_counts.append(rhs_send)

            # Create send buffer
            lhs_snd_buffers.append(np.zeros((self.ndims * self.nvars * lhs_send), dtype=self._be.fpdtype))
            rhs_snd_buffers.append(np.zeros((self.ndims * self.nvars * rhs_send), dtype=self._be.fpdtype))


        # Buffer for final unpacked data
        local_lhs_interp = np.zeros((self.ndims, self.nvars, self.ninterfpts_lhs), dtype=self._be.fpdtype)
        local_rhs_interp = np.zeros((self.ndims, self.nvars, self.ninterfpts_rhs), dtype=self._be.fpdtype)

        # Send/Recv interpolated data
        mpi_requests = []
        for rank in range(0, self.comm.size):
            if rank == self.comm.rank:
                # Copy interpolated data that will stay locally
                for iidx in range(0, len(lhs2rhs_iinfo)):
                    if lhs2rhs_iinfo[iidx][1] == self.comm.rank:
                        local_rhs_interp[:,:,lhs2rhs_iinfo[iidx][2]] = idata_for_remote_rhs[:,:,iidx]
                for iidx in range(0, len(rhs2lhs_iinfo)):
                    if rhs2lhs_iinfo[iidx][1] == self.comm.rank:
                        local_lhs_interp[:,:,rhs2lhs_iinfo[iidx][2]] = idata_for_remote_lhs[:,:,iidx]
            else:
                # Non-blocking receive
                if lhs_rcv_counts[rank] > 0:
                    mpi_requests.append(self.comm.Irecv(lhs_rcv_buffers[rank], rank, self.LHS_GRAD_MPI_TAG))
                if rhs_rcv_counts[rank] > 0:
                    mpi_requests.append(self.comm.Irecv(rhs_rcv_buffers[rank], rank, self.RHS_GRAD_MPI_TAG))

                # Pack data to send to this rank and non-blocking send
                if lhs_snd_counts[rank] > 0:
                    self._pack_send_grad_buffer(rhs2lhs_iinfo, idata_for_remote_lhs, lhs_snd_buffers[rank], rank)
                    mpi_requests.append(self.comm.Isend(lhs_snd_buffers[rank], rank, self.LHS_GRAD_MPI_TAG))
                if rhs_snd_counts[rank] > 0:
                    self._pack_send_grad_buffer(lhs2rhs_iinfo, idata_for_remote_rhs, rhs_snd_buffers[rank], rank)
                    mpi_requests.append(self.comm.Isend(rhs_snd_buffers[rank], rank, self.RHS_GRAD_MPI_TAG))
        
        # Wait on non blocking comms
        for req in mpi_requests:
            req.Wait()
            req.free()
        
        # Unpack received data and update PyFR matrices
        if self.ninters_lhs:
            self._unpack_recv_grad_buffers(lhs_rcv_buffers, local_lhs_interp, self.lhs_pts_rhs_fidx)
            self._vect_lhs_interp.set(local_lhs_interp)
        
        if self.ninters_rhs:
            self._unpack_recv_grad_buffers(rhs_rcv_buffers, local_rhs_interp, self.rhs_pts_lhs_fidx)
            self._vect_rhs_interp.set(local_rhs_interp)
        
        self._grad_comm_time += time.time() - tstart
        if self._t_counter % 1000 == 0:
            print(f'{self._grad_comm_time}')
    

