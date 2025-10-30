import numpy as np

from pyfr.cache import memoize
from pyfr.shapes import BaseShape
from pyfr.quadrules import get_quadrule
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


class VTKSpanAvgWriter(BaseVTKWriter):
    type = 'spanavg'
    output_curved = True
    output_partition = False

    def _load_hex(self):
        if 'hex' not in self.mesh.spts:
            return None, None, None

        nspts, neles = self.mesh.spts['hex'].shape[:2]

        # 2D number of shape pts        
        if nspts == 8:
            nspts2d = 4
        elif nspts == 27:
            nspts2d = 9
        else:
            raise RuntimeError('Unrecognised nspts')
        
        # Get Quad rules for hex elements
        shapecls = subclass_where(BaseShape, name='hex')
        shape = shapecls(nspts, self.cfg)
        order = self.cfg.getint('solver', 'order')
        # Get physical location of each soln point
        ploc = shape.sbasis.nodal_basis_at(shape.upts).astype(self.dtype) @ self.mesh.spts['hex'].reshape(nspts, -1)
        ploc = ploc.reshape(shape.nupts, -1, self.ndims)
        ploc = np.swapaxes(ploc, 0, 1)
        # Get weights at each soln point (but the line shape weights as we're integrating along separate lines, not volumes)
        line = get_quadrule('line', self.cfg.get(f'solver-elements-{shape.name}', 'soln-pts'), order + 1)
        # Get number of quad solution points
        quadnupts = (order + 1)**2
        # Swap axis of soln to match ploc
        soln = self.soln['hex'].swapaxes(0, 1).swapaxes(0, 2)

        nvals = soln.shape[2]
        curved = self.mesh.spts_curved['hex']
        _ploc2d = np.zeros((quadnupts, 2), dtype=self.dtype)
        _soln2d = np.zeros((quadnupts, nvals), dtype=self.dtype)
        _mesh2d = np.zeros((nspts2d, 2), dtype=self.dtype)
        elements2d = {}
        centreAvgDP = 6
        ploc2dTol = 1e-6
        # Iterate over each element and reduce to 2D quad, and add to map in order to reduce a stack of quads onto a single quad
        for _ploc, _soln, _mesh, _curved in zip(ploc, soln, self.mesh.spts['hex'].swapaxes(0, 1), curved):
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
                if _curved:
                    acc[4] += 1
            else:
                acc = [_ploc2d, _soln2d, 1, _mesh2d, 1 if _curved else 0]
            elements2d[plockey] = acc

        # Build new arrays
        nelem2d = len(elements2d)
        ploc2d = np.zeros((nelem2d, quadnupts, 2), dtype=self.dtype)
        soln2d = np.zeros((nelem2d, quadnupts, nvals), dtype=self.dtype)
        mesh2d = np.zeros((nelem2d, nspts2d, 2), dtype=self.dtype)
        curved = []
        elemind = 0
        for _plockey in elements2d:
            ploc2d[elemind] = elements2d[_plockey][0]
            soln2d[elemind] = elements2d[_plockey][1] / self.dtype(elements2d[_plockey][2])
            mesh2d[elemind] = elements2d[_plockey][3]
            curved.append(elements2d[_plockey][4] != 0)
            elemind += 1
        # Reorder axes in the original way for solution
        soln2d = soln2d.swapaxes(0, 1).swapaxes(1, 2)
        # Reorder the axes in the original way for the mesh
        mesh2d = mesh2d.swapaxes(0, 1)

        return mesh2d, soln2d, curved
    
    def _load_pri(self):
        if 'pri' not in self.mesh.spts:
            return None, None, None

        nspts, neles = self.mesh.spts['pri'].shape[:2]

        # 2D number of shape pts        
        if nspts == 6:
            nspts2d = 3
        elif nspts == 18:
            nspts2d = 6
        else:
            raise RuntimeError('Unrecognised nspts')
        
        # Get Quad rules for hex elements
        shapecls = subclass_where(BaseShape, name='pri')
        shape = shapecls(nspts, self.cfg)
        order = self.cfg.getint('solver', 'order')
        # Get physical location of each soln point
        ploc = shape.sbasis.nodal_basis_at(shape.upts).astype(self.dtype) @ self.mesh.spts['pri'].reshape(nspts, -1)
        ploc = ploc.reshape(shape.nupts, -1, self.ndims)
        ploc = np.swapaxes(ploc, 0, 1)
        # Get weights at each soln point (but the line shape weights as we're integrating along separate lines, not volumes)
        if self.cfg.get(f'solver-elements-{shape.name}', 'soln-pts') != 'williams-shunn~gauss-legendre':
            raise Exception('Span averaging on prisms only supports the williams-shunn~gauss-legendre solution points')
        line = get_quadrule('line', 'gauss-legendre', order + 1)
        # Get number of tri solution points
        trinupts = (order + 1)*((order + 1) + 1) // 2
        # Swap axis of soln to match ploc
        soln = self.soln['pri'].swapaxes(0, 1).swapaxes(0, 2)
        # Get new arrays
        nvals = soln.shape[2]
        curved = self.mesh.spts_curved['pri']
        _ploc2d = np.zeros((trinupts, 2), dtype=self.dtype)
        _soln2d = np.zeros((trinupts, nvals), dtype=self.dtype)
        _mesh2d = np.zeros((nspts2d, 2), dtype=self.dtype)
        elements2d = {}
        centreAvgDP = 6
        ploc2dTol = 1e-6
        # Iterate over each element and reduce to 2D tri, also add to map to reduce a stack of tris to a single tri
        for _ploc, _soln, _mesh, _curved in zip(ploc, soln, self.mesh.spts['pri'].swapaxes(0, 1), curved):
            # TODO will this always be the case for shape points?
            for i in range(0, nspts2d):
                _mesh2d[i][0] = _mesh[i][0]
                _mesh2d[i][1] = _mesh[i][1]
            # Integrate within element to spanwise average within an element
            for i in range(0, trinupts):
                _ploc2d[i][0] = _ploc[i][0]
                _ploc2d[i][1] = _ploc[i][1]
                for v in range(0, nvals):
                    for j in range(0, order + 1):
                        _soln2d[i][v] += _soln[i + j * trinupts][v] * line.wts[j]
                    # Line weights integrate [-1,1], so need to divide by 2
                    _soln2d[i][v] *= 0.5
            
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
                        print('pri')
                        print('***')
                acc[2] += 1
                if _curved:
                    acc[4] += 1
            else:
                acc = [_ploc2d, _soln2d, 1, _mesh2d, 1 if _curved else 0]
            elements2d[plockey] = acc

        # Build new arrays
        nelem2d = len(elements2d)
        ploc2d = np.zeros((nelem2d, trinupts, 2), dtype=self.dtype)
        soln2d = np.zeros((nelem2d, trinupts, nvals), dtype=self.dtype)
        mesh2d = np.zeros((nelem2d, nspts2d, 2), dtype=self.dtype)
        curved = []
        elemind = 0
        for _plockey in elements2d:
            ploc2d[elemind] = elements2d[_plockey][0]
            soln2d[elemind] = elements2d[_plockey][1] / self.dtype(elements2d[_plockey][2])
            mesh2d[elemind] = elements2d[_plockey][3]
            curved.append(elements2d[_plockey][4] != 0)
            elemind += 1
        # Reorder axes in the original way for solution
        soln2d = soln2d.swapaxes(0, 1).swapaxes(1, 2)
        # Reorder the axes in the original way for the mesh
        mesh2d = mesh2d.swapaxes(0, 1)

        return mesh2d, soln2d, curved

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        hexmesh2d, hexsoln2d, hexcurved = self._load_hex()

        primesh2d, prisoln2d, pricurved = self._load_pri()

        class SpanMeshAdapter:
            etypes = []
            eidxs = {}
            spts = {}
            spts_curved = {}

        _mesh2d = SpanMeshAdapter()

        if hexmesh2d is not None:
            _mesh2d.etypes.append('quad')
            _mesh2d.eidxs['quad'] = np.array([i for i in range(0, hexmesh2d.shape[1])])
            _mesh2d.spts['quad'] = hexmesh2d
            _mesh2d.spts_curved['quad'] = hexcurved
        
        if primesh2d is not None:
            _mesh2d.etypes.append('tri')
            _mesh2d.eidxs['tri'] = np.array([i for i in range(0, primesh2d.shape[1])])
            _mesh2d.spts['tri'] = primesh2d
            _mesh2d.spts_curved['tri'] = pricurved

        self.mesh = _mesh2d

        if hexsoln2d is not None:
            self.soln['quad'] = hexsoln2d
            self.soln['quad-parts'] = np.ones((hexmesh2d.shape[1]))
        
        if prisoln2d is not None:
            self.soln['tri'] = prisoln2d
            self.soln['tri-parts'] = np.ones((primesh2d.shape[1]))

        # Update cfg for 2D
        if hexmesh2d is not None:
            self.cfg.set('solver-elements-quad', 'soln-pts', self.cfg.get(f'solver-elements-hex', 'soln-pts'))
            self.cfg.rename_section('solver-elements-hex', 'z1')
        
        if primesh2d is not None:
            self.cfg.set('solver-elements-tri', 'soln-pts', 'williams-shunn')
            self.cfg.rename_section('solver-elements-pri', 'z2')

        self.ndims = 2

        self.einfo = [(etype, self.soln[etype].shape[2])
                      for etype in self.mesh.eidxs]

    @memoize
    def _opmats(self, etype, cfg):
        # Shape
        shapecls = subclass_where(BaseShape, name=etype)

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.etypes_div[etype])
        nsvpts = len(svpts)

        # Basis
        basis = shapecls(len(self.mesh.spts[etype]), cfg)

        if etype != 'pyr' and self.ho_output:
            svpts = [svpts[i] for i in self._nodemaps[etype, nsvpts]]

        mesh_op = basis.sbasis.nodal_basis_at(svpts)
        soln_op = basis.ubasis.nodal_basis_at(svpts)

        return mesh_op, soln_op

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)
        soln = self.soln[etype].swapaxes(0, 1).astype(self.dtype)
        curved = self.mesh.spts_curved[etype]

        # Extract the partition number information
        part = self.soln[f'{etype}-parts']

        # Generate the interpolation operator matrices
        mesh_vtu_op, soln_vtu_op = self._opmats(etype, self.cfg)

        # Calculate node locations of VTU elements
        vpts = interpolate_pts(mesh_vtu_op, spts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Pre-process the solution
        soln = self._pre_proc_fields(soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = interpolate_pts(soln_vtu_op, soln)

        return vpts, vsoln, curved, part
