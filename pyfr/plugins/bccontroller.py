import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin
from rtree.index import Index, Property
from pyfr.quadrules import get_quadrule

# Largely taken from SamplerPlugin
from pyfr.plugins.sampler import _closest_pts, _plocs_to_tlocs
class PtSampler():
    def __init__(self, intg, cfgsect):
        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars
        self.cfg = intg.cfg
        # Underlying elements class
        self.elementscls = intg.system.elementscls
        # List of points to be sampled (will only be 1 point but kept as list to reuse SamplerPlugin code)
        self.pts = [self.cfg.getliteral(cfgsect, 'samp-pt')]
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each sample point
        if rank == root:
            ptsrank = []

        # Sample points we're responsible for, grouped by element type
        elepts = [[] for i in range(len(intg.system.ele_map))]

        # Search locations in transformed and physical space
        tlocs, plocs = self._search_pts(intg)

        # For each sample point find our nearest search location
        closest = _closest_pts(plocs, self.pts)

        # Process these points
        for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
            # Reduce over the distance
            _, mrank = comm.allreduce((dist, rank), op=mpi.MINLOC)

            # If we have the closest point then save the relevant info
            if rank == mrank:
                elepts[etype].append((i, eidx, tlocs[etype][uidx]))

            # Note what rank is responsible for the point
            if rank == root:
                ptsrank.append(mrank)

        # Refine
        self._ourpts = ourpts = self._refine_pts(intg, elepts)

        # Send the refined sample locations to the root rank
        ptsplocs = comm.gather([pl for et, ei, pl, op in ourpts], root=root)

        if rank == root:
            nvars = self.nvars

            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), self.nvars))

            # Tally up how many points each rank is responsible for
            nptsrank = [len(ploc) for ploc in ptsplocs]

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array(nptsrank, dtype=np.int32)
            ptsdisps = np.cumsum([0] + nptsrank[:-1], dtype=np.int32)

            # Apply the displacements to each ranks points
            miters = [enumerate(ploc, start=pdisp)
                      for ploc, pdisp in zip(ptsplocs, ptsdisps)]

            # With this form the final point (offset, location) list
            self._ptsinfo = [next(miters[pr]) for pr in ptsrank]

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))
        else:
            self._ptsrecv = None
        
        self.rank_with_pt = 0
        if rank == root:
            self.rank_with_pt = ptsrank[0]
        comm.bcast(self.rank_with_pt, root=root)
        
    def _search_pts(self, intg):
        tlocs, plocs = [], []

        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        for etype, eles in intg.system.ele_map.items():
            pts = get_quadrule(etype, qrule_map[etype], eles.basis.nupts).pts

            tlocs.append(pts)
            plocs.append(eles.ploc_at_np(pts).swapaxes(1, 2))

        return tlocs, plocs

    def _refine_pts(self, intg, elepts):
        elelist = intg.system.ele_map.values()
        ptsinfo = []

        # Loop over all the points for each element type
        for etype, (eles, epts) in enumerate(zip(elelist, elepts)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles.eles[:, eidx, :]
            plocs = [self.pts[i] for i in idx]

            # Use Newton's method to find the precise transformed locations
            ntlocs, nplocs = _plocs_to_tlocs(eles.basis.sbasis, spts, plocs,
                                             tlocs)

            # Form the corresponding interpolation operators
            intops = eles.basis.ubasis.nodal_basis_at(ntlocs)

            # Append to the point info list
            ptsinfo.extend(
                (*info, etype) for info in zip(idx, eidx, nplocs, intops)
            )

        # Sort our info array by its original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(etype, *info) for idx, *info, etype in ptsinfo]

    def _process_samples(self, samps):
        samps = np.array(samps)
        # Convert to primitive form
        samps = self.elementscls.con_to_pri(samps.T, self.cfg)
        samps = np.array(samps).T
        return np.ascontiguousarray(samps, dtype=float)
    
    def __call__(self, intg):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the solution matrices
        solns = intg.soln

        # Perform the sampling and interpolation
        samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
        samples = self._process_samples(samples)

        # Only 1 sample point
        comm.bcast(samples, root=self.rank_with_pt)

        # Rho, u, v, (w,) p
        return samples[0]

class BcControllerPlugin(BaseSolverPlugin):
    name = 'bccontroller'
    systems = ['euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Constants from config file
        self.consts = self.cfg.items_as('constants', float)

        # Point sampler
        self.ptsampler = PtSampler(intg, cfgsect)

        # Cumulative error
        self.cerr = 0.0

        # Target Mach number
        self.targetmach = self.cfg.getfloat(cfgsect, 'target-mach')

        # PI controller parameters
        self.kp = self.cfg.getfloat(cfgsect, 'kp')
        self.ki = self.cfg.getfloat(cfgsect, 'ki')
        self.propdelay = self.cfg.getfloat(cfgsect, 'propergation-delay')

        # Set first p value in BC mako kernel
        self.p = self.cfg.getfloat(cfgsect, 'p')
        self.lastupdate = intg.tcurr
        intg.system.update_kernel_extern('c_p', self.p)

    def __call__(self, intg):
        if intg.tcurr < self.lastupdate + self.propdelay:
            intg.system.update_kernel_extern('c_p', self.p)
            return
        # Get Mach number
        mach = self.mach_at_pt(intg)

        # PI controller
        err = self.targetmach - mach
        factor = 1.0 + self.kp * err + self.ki * self.cerr
        print(f'Error: {err} Factor: {factor}')
        self.p = self.p * factor
        self.cerr = self.cerr + err
        self.lastupdate = intg.tcurr

        intg.system.update_kernel_extern('c_p', self.p)
    
    def mach_at_pt(self, intg):
        primitives = self.ptsampler(intg)
        # Speed of sound
        c = np.sqrt(self.consts['gamma'] * primitives[-1] / primitives[0])
        # Velocity magnitude
        vmag = 0.0
        if self.ndims == 2:
            vmag = np.sqrt(primitives[1]**2 + primitives[2]**2)
        else:
            vmag = np.sqrt(primitives[1]**2 + primitives[2]**2 + primitives[3]**2)
        # Mach number
        return vmag / c