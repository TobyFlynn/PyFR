from pyfr.writers.vtk import VTKWriter
from pyfr.quadrules import get_quadrule
from pyfr.util import memoize

import numpy as np

class SpanAvgVTKWriter(VTKWriter):
    # Supported file types and extensions
    name = 'spanavgvtk'
    extn = ['.span']

    def __init__(self, args):
        super().__init__(args)

        print("WARNING: This spanwise averaging assumes that the Z dimension layers of the mesh are evenly spaced.")
        print("WARNING: The mesh and solution must only have 1 partition.")
        print("WARNING: Only hex meshes are currently supported.")

        nspts, neles = self.mesh['spt_hex_p0'].shape[:2]

        if 'soln_hex_p0' in self.soln:
            solnkey = 'soln_hex_p0'
            solnkey2d = 'soln_quad_p0'
        else:
            solnkey = 'tavg_hex_p0'
            solnkey2d = 'tavg_quad_p0'

        # TODO make this better
        if nspts == 8:
            nspts2d = 4
        elif nspts == 27:
            nspts2d = 9
        else:
            raise RuntimeError('Unrecognised nspts')

        # Get Quad rules for hex elements
        shape = self._get_shape('hex', nspts)
        order = self.cfg.getint('solver', 'order')
        # Get physical location of each soln point
        ploc = shape.sbasis.nodal_basis_at(shape.upts).astype(self.dtype) @ self.mesh['spt_hex_p0'].reshape(nspts, -1)
        ploc = ploc.reshape(shape.nupts, -1, self.ndims)
        ploc = np.swapaxes(ploc, 0, 1)
        # Get weights at each soln point (but the line shape weights as we're integrating along separate lines, not volumes)
        line = get_quadrule('line', self.cfg.get(f'solver-elements-{shape.name}', 'soln-pts'), order + 1)
        # Get number of quad solution points
        quadnupts = (order + 1)**2
        # Swap axis of soln to match ploc
        soln = self.soln[solnkey].swapaxes(0, 1).swapaxes(0, 2)
        # Get new arrays
        nvals = soln.shape[2]
        ploc2d = np.zeros((neles, quadnupts, 2), dtype=self.dtype)
        soln2d = np.zeros((neles, quadnupts, nvals), dtype=self.dtype)
        mesh2d = np.zeros((neles, nspts2d, 2), dtype=self.dtype)
        # Iterate over each element and reduce to 2D quad
        for _ploc, _soln, _mesh, _ploc2d, _soln2d, _mesh2d in zip(ploc, soln, self.mesh['spt_hex_p0'].swapaxes(0, 1), ploc2d, soln2d, mesh2d):
            # TODO will this always be the case for shape points?
            for i in range(0, nspts2d):
                _mesh2d[i][0] = _mesh[i][0]
                _mesh2d[i][1] = _mesh[i][1]
            # Integrate within element to spanwise average within an element
            for i in range(0, quadnupts):
                _ploc2d[i][0] = _ploc[i][0]
                _ploc2d[i][1] = _ploc[i][1]
                for v in range(0, nvals):
                    for j in range(0, order + 1):
                        _soln2d[i][v] += _soln[i + j * quadnupts][v] * line.wts[j]
                    # Line weights integrate [-1,1], so need to divide by 2
                    _soln2d[i][v] *= 0.5
        # Reduce stacks of 2D quads to a single quad
        # *** This assumes evenly spaced Z dimension layers in the mesh!!! ***
        elements2d = {}
        centreAvgDP = 6
        ploc2dTol = 1e-6
        for _ploc2d, _soln2d, _mesh2d in zip(ploc2d, soln2d, mesh2d):
            # Key is average point rounded to 6 decimal places 
            plockey = (np.round(np.sum(_ploc2d[:,0]) / len(_ploc2d), centreAvgDP), np.round(np.sum(_ploc2d[:,1]) / len(_ploc2d), centreAvgDP))
            if plockey in elements2d:
                acc = elements2d[plockey]
                for ptInd in range(0, len(_ploc2d)):
                    found = False
                    for accInd in range(0, len(_ploc2d)):
                        if abs(_ploc2d[ptInd,0] - acc[0][accInd,0]) < ploc2dTol and abs(_ploc2d[ptInd,1] - acc[0][accInd,1]) < ploc2dTol:
                            acc[1][accInd] += _soln2d[ptInd]
                            found = True
                            break
                    if not found:
                        print('Did not find a matching solution point!')
                        print(f'Point: {_ploc2d[ptInd]}')
                        print(f'Point key: {plockey}')
                        print(acc[0])
                        print('***')
                acc[2] += 1
            else:
                acc = [_ploc2d, _soln2d, 1, _mesh2d]
            elements2d[plockey] = acc
        # Build new arrays
        nelem2d = len(elements2d)
        ploc2d = np.zeros((nelem2d, quadnupts, 2), dtype=self.dtype)
        soln2d = np.zeros((nelem2d, quadnupts, nvals), dtype=self.dtype)
        mesh2d = np.zeros((nelem2d, nspts2d, 2), dtype=self.dtype)
        elemind = 0
        for _plockey in elements2d:
            ploc2d[elemind] = elements2d[_plockey][0]
            soln2d[elemind] = elements2d[_plockey][1] / self.dtype(elements2d[_plockey][2])
            mesh2d[elemind] = elements2d[_plockey][3]
            elemind += 1
        # Reorder axes in the original way for solution
        soln2d = soln2d.swapaxes(0, 1).swapaxes(1, 2)
        # Reorder the axes in the original way for the mesh
        mesh2d = mesh2d.swapaxes(0, 1)
        # Create mesh/soln adapter (can just be a dictonary) for the base VTK writer
        self.mesh = {'spt_quad_p0': mesh2d}
        self.soln = {solnkey2d: soln2d}
        # Now reconstruct the other datastructures/variables that the VTK writer expects from Base
        self.mesh_inf = {'spt_quad_p0': ('quad', (nspts2d, nelem2d, 2))}
        self.soln_inf = {solnkey2d: ('quad', (quadnupts, nvals, nelem2d))}
        self.ndims = 2
        # Now reconstruct the other datastructures/variables that the VTK writer expects from VTK
        self._pre_proc_fields = self._pre_proc_fields_scal
        self._post_proc_fields = self._post_proc_fields_scal
        self._soln_fields = self.stats.get('data', 'fields').split(',')
        self._vtk_vars = [(k, [k]) for k in self._soln_fields]
        self.tcurr = None
        # Update cfg for 2D
        self.cfg.set('solver-elements-quad', 'soln-pts', self.cfg.get(f'solver-elements-{shape.name}', 'soln-pts'))
        self.cfg.rename_section('solver-elements-hex', 'z')


