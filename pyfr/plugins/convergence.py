import numpy as np
import csv
from pyfr.plugins.base import BaseSolnPlugin
from pyfr.mpiutil import get_comm_rank_root

# Checks if we have reached convergence for the OGV test case
# Currently assumes a fixed dt
class ConvergencePlugin(BaseSolnPlugin):
    name = 'convergence'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.dtcheck = self.cfg.getfloat(cfgsect, 'dt-check')
        self.tmax_transient = self.cfg.getfloat(cfgsect, 'tmax-transient')
        self.ciTarget = self.cfg.getfloat(cfgsect, 'ci-target')
        self.tTol = self.cfg.getfloat(cfgsect, 'terminate-tol', 1e-7)

        self.mp1File = self.cfg.get(self.cfgsect, 'mp1-file')
        self.mp2File = self.cfg.get(self.cfgsect, 'mp2-file')
        self.outFile = self.cfg.get(self.cfgsect, 'out-file')

        self._started = False
        self._terminate_time = None
    
    def _get_mixed_out_from_int_vals(self, csvrow):
        mDot = float(csvrow['int-mdot'])
        x = float(csvrow['int-x1']) + float(csvrow['int-x2'])
        y = float(csvrow['int-y1'])
        z = float(csvrow['int-z1'])
        e = (self.gamma * 8.31446261815324 / (self.gamma - 1.0)) * float(csvrow['int-e1']) + 0.5 * float(csvrow['int-e2'])
        area = float(csvrow['int-vol'])
        # Calculate static pressure
        Q = (1.0 / (mDot * mDot)) * (area * area - (2.0 * self.gamma / (self.gamma - 1.0)) * area * area)
        L = (2.0 / (mDot * mDot)) * ((self.gamma / (self.gamma - 1.0)) * x * area - x * area)
        C = (1.0 / (mDot * mDot)) * (x * x + y * y + z * z) - 2.0 * e / mDot
        p = (-L - np.sqrt(L * L - 4.0 * Q * C)) / (2.0 * Q)
        # Get other variables from this
        vx = (x - p * area) / mDot
        rho = mDot / (area * vx)
        t = p / (8.31446261815324 * rho)
        tt = ((self.gamma - 1.0) / (self.gamma * 8.31446261815324)) * (e / mDot)
        pt = p * np.power(tt / t, self.gamma / (self.gamma - 1.0))
        return p, pt

    def _load_obj_time_series(self):
        # Load time series from MP1
        time = []
        p1 = []
        pt1 = []
        with open(self.mp1File) as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                p, pt = self._get_mixed_out_from_int_vals(row)
                _time = float(row['t'])
                time.append(_time)
                p1.append(p)
                pt1.append(pt)
        # Load time series from MP2
        p2 = []
        pt2 = []
        with open(self.mp2File) as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                p, pt = self._get_mixed_out_from_int_vals(row)
                p2.append(p)
                pt2.append(pt)
        # Calculate time series of objective function/exit angle
        objFun = []
        for i in range(0, len(time)):
            objFun.append((pt1[i] - pt2[i]) / (pt1[i] - p1[i]))
        return time, objFun

    # Detecting transient
    def _get_transient_ind(self, array, time, tmax):
        def truncacted_mean(array, d):
            num = len(array) - d
            total = sum(array[d:])
            return total / float(num)
        def mser_function(array, d):
            num = len(array) - d
            total = 0.0
            truncAvg = truncacted_mean(array, d)
            for i in range(d, len(array)):
                total = total + (array[i] - truncAvg)**2
            return total / float(num**2)
        mser_array = []
        for i in range(0, len(array)):
            if time[i] > tmax:
                break
            mser_array.append(mser_function(array, i))
        return min(range(len(mser_array)), key=mser_array.__getitem__)
    
    # Calculate 95% confidence interval (as a percentage)
    def _calc_CI(array):
        # Helper functions
        def mean_N(N, array):
            return sum(array[:N]) / N
        def var_N(N, array):
            meanN = mean_N(N, array)
            acc = 0.0
            for i in range(0, N):
                acc = acc + (array[i] - meanN)**2
            return acc / N
        def rho_N_tau(N, tau, array):
            sumNumerator = 0.0
            meanN = mean_N(N, array)
            for i in range(tau, N):
                sumNumerator = sumNumerator + (array[i] - meanN) * (array[i - tau] - meanN)
            varN = var_N(N, array)
            return (sumNumerator / (N - tau)) / varN
        def D_N(N, array):
            acc = 0.0
            for i in range(1, N):
                rho = rho_N_tau(N, i, array)
                if rho < 0.0:
                    break
                acc = acc + rho
            return 1.0 + 2.0 * acc
        # Actual function body
        DN = D_N(len(array), array)
        varN = var_N(len(array), array)
        ci = 1.96 * np.sqrt(DN * varN / len(array))
        mean = sum(array) / len(array)
        return 100.0 * ci / mean

    def __call__(self, intg):
        # If we are not supposed to be checking convergence yet then return
        if intg.tcurr < self.tstart:
            return
        
        if not self._started:
            self.tcheck_last = intg.tcurr
            self._started = True
        
        docheck = intg.tcurr - self.tcheck_last >= self.dtcheck - self.tol

        if docheck:
            # Reset time that we last checked convergence
            self.tcheck_last = intg.tcurr

            comm, rank, root = get_comm_rank_root()

            if rank == root:
                # Get objective function as a time series
                time, objFun = self._load_obj_time_series()

                # Remove transient
                transientInd = self._get_transient_ind(objFun, time, self.tmax_transient)
                objFun = objFun[transientInd:]
                time = time[transientInd:]

                # Get confidence interval on this
                ci = self._calc_CI()

                # Decide whether to terminate
                if ci < self.ciTarget:
                    with open(self.outFile, 'w') as f:
                        f.write(f'Transient time: {time[0]}\n')
                        f.write(f'Mean: {sum(objFun) / len(objFun)}\n')
                        f.write(f'CI: {ci}\n')
                    self._terminate_time = intg.tcurr + self.tTol
        
        if self._terminate_time is not None:
            if self._terminate_time > intg.tcurr:
                # Terminate the simulation
                raise RuntimeError(f'Early termination of simulation due to convergence criteria being met.')
