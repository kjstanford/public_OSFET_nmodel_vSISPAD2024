from helper_funs import *
import re
import ast

def generate_figname():
    ii = 0
    pwdpath = os.path.dirname(os.path.realpath(__file__))
    while os.path.isfile(os.path.join(pwdpath, f'temp{ii}.png')):
        ii = ii + 1
    return f'temp{ii}.png'

class OSFET_numerical_model_v1_0_0():
    """
    This model has been submitted for SISPAD 2024 and used for VLSI 2024
    """
    def __init__(self, params):
        """
        Model parameters defined here
        """
        self.mu_band = params[
            'mu_band']  # band mobility ( constant mobility assumed for extended states above the band edge ) in m^2.V^-1.s^-1
        self.T = params['T']  # device temperature ( no self heating included) in K
        self.tCH = params['tCH']  # OS channel thickness in m
        self.tDE = params['tDE']  # gate dielectric thickness in m
        self.me = params['me_factor'] * mo  # effective mass in kg
        self.kDE = params['kDE']  # relative permittivity of gate dielectric
        self.kCH = params['kCH']  # relative permittivity of OS channel
        self.phiM = params['phiM']  # gate metal workfunction in eV
        self.chiS = params['chiS']  # OS electron affinity in eV
        # self.Vds = params['Vds'] # drain voltage w.r.t. source in V
        self.L = params['L']  # channel length in m
        self.W = params['W']  # channel width in m
        self.NCH = params['NCH']  # fixed donor channel doping in m^-3
        self.trap_list_3D = [] if not 'trap_list_3D' in params else params[
            'trap_list_3D']  # 3D trap density in the OS bulk ( in cm^-3.eV^-1 )
        self.trap_list_2D = [] if not 'trap_list_2D' in params else params[
            'trap_list_2D']  # 2D trap density at some region inside gate dielectric or OS ( in cm^-2.eV^-1 )
        self.fixed_charge_list_2D = [] if not 'fixed_charge_list_2D' in params else params[
            'fixed_charge_list_2D']  # 2D fixed charge density at some region inside gate dielectric or OS ( in cm^-2 )
        self.band_tail_list_2D = [] if not 'band_tail_list_2D' in params else params[
            'band_tail_list_2D']  # exponential band tail states attached to extended states, below mobility edge ( in cm^-2.eV^-1 )
        self.HfO2_offset = params['HfO2_offset']  # offset between conduction bands of HfO2 and OS in eV
        """
        Other variables
        """
        self.phiT = kB * self.T
        self.LD = ((2 * self.kCH * epso * self.phiT) / (q * self.NCH)) ** (
                    1 / 2)  # debye length in m (only w.r.t the fixed donor channel doping NCH)
        self.CDE = epso * self.kDE / self.tDE  # gate dielectric capacitance in F.m^-2
        self.N3D = 2 * (2 * pi * self.me * q * kB * self.T / h ** 2) ** 1.5  # effective 3D DoS density in m^-3
        self.N2D = self.me * q * kB * self.T / (pi * hred ** 2)  # effective 2D DoS density in m^-2
        self.num_levels = 10  # no. of smallest eigen-values

        """
        Model hyper-parameters defined here
        """
        if 'os_step_size' in params:
            self.OS_step_size = params['os_step_size']
        else:
            self.OS_step_size = 5e-11
        # self.num_levels = params['num_levels']

        """
        Grid definition for the solver
        """
        self.OS_channel_grid = self.tDE + np.arange(0, self.tCH + self.OS_step_size, self.OS_step_size)
        xgrid = np.insert(self.OS_channel_grid, 0, 0)

        # for trap_2D in self.trap_list_2D:
        #     xtrap2D = trap_2D['xtrap2D']
        #     if not xtrap2D in xgrid:
        #         xgrid = np.insert(xgrid, 0, xtrap2D)
        #         xgrid.sort(kind='heapsort')

        # for fixed_2D in self.fixed_charge_list_2D:
        #     xfixed = fixed_2D['xfixed']
        #     if not xfixed in xgrid:
        #         xgrid = np.insert(xgrid, 0, xfixed)
        #         xgrid.sort(kind='heapsort')

        self.xgrid = xgrid  # grid along x-direction for schrodinger-poisson solving
        self.grid_size = len(self.xgrid)

        self.kfun = lambda x: self.kDE if x <= self.tDE else self.kCH  # dielectric constant as function of x
        self.phigrid = np.ones(np.shape(self.xgrid))

        """
        Compute FEM-based matrix for solving poisson equation
        """
        self.sf = ((self.xgrid[self.grid_size - 1] - self.xgrid[self.grid_size - 2]) ** 2) / self.kfun(
            self.xgrid[self.grid_size - 1])
        # print(self.sf)

        self.poisson_mat = np.diag(np.zeros(np.shape(self.xgrid)))
        self.poisson_mat[0, 0] = 1
        for ii in range(1, self.grid_size - 1):
            self.poisson_mat[ii, ii - 1] = self.sf * (
                        self.kfun(self.xgrid[ii]) / (self.xgrid[ii] - self.xgrid[ii - 1])) * (
                                                       2 / (self.xgrid[ii + 1] - self.xgrid[ii - 1]))
            self.poisson_mat[ii, ii] = self.sf * (
                        (self.kfun(self.xgrid[ii + 1]) / (self.xgrid[ii + 1] - self.xgrid[ii])) + (
                            self.kfun(self.xgrid[ii]) / (self.xgrid[ii] - self.xgrid[ii - 1]))) \
                                       * (-2 / (self.xgrid[ii + 1] - self.xgrid[ii - 1]))
            self.poisson_mat[ii, ii + 1] = self.sf * (
                        self.kfun(self.xgrid[ii + 1]) / (self.xgrid[ii + 1] - self.xgrid[ii])) * (
                                                       2 / (self.xgrid[ii + 1] - self.xgrid[ii - 1]))
        self.poisson_mat[self.grid_size - 1, self.grid_size - 1] = self.sf * (
                    self.kfun(self.xgrid[self.grid_size - 1]) / (
                        self.xgrid[self.grid_size - 1] - self.xgrid[self.grid_size - 2])) * (-1 / (
                    self.xgrid[self.grid_size - 1] - self.xgrid[self.grid_size - 2]))
        self.poisson_mat[self.grid_size - 1, self.grid_size - 2] = self.sf * (
                    self.kfun(self.xgrid[self.grid_size - 1]) / (
                        self.xgrid[self.grid_size - 1] - self.xgrid[self.grid_size - 2])) * (1 / (
                    self.xgrid[self.grid_size - 1] - self.xgrid[self.grid_size - 2]))

    def poisson_solve(self, Vgs, Vy):
        """
        Initialize phigrid
        """
        self.phigrid[0] = Vgs - (self.phiM - self.chiS)
        self.phigrid[1:] = self.phigrid[1:] * min(0, self.phigrid[0]) + self.phiT

        Ms = sparse.csr_matrix(self.poisson_mat)  # sparse matrix version of poisson_mat

        """
        Forcing function of the poisson equation calculated from the charges
        """
        def FFsolve(phigrid):
            Qfree = lambda ii: 0 if self.xgrid[ii] <= self.tDE else q * self.N3D * FD_int_3D(
                (phigrid[ii] - Vy) / self.phiT)
            Qdop = lambda ii: 0 if self.xgrid[ii] <= self.tDE else -1 * q * self.NCH
            Qfixed = lambda ii: sum([0 if self.xgrid[ii] != fixed_2D['xfixed'] \
                                         else q * fixed_2D['Nfixed'] * (2 if ii < self.grid_size - 1 else 1) / (
                        self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[ii - 1]) \
                                     for fixed_2D in self.fixed_charge_list_2D])
            Qtrap2D = lambda ii: sum([0 if self.xgrid[ii] != trap_2D['xtrap2D'] \
                                          else q * gaussian_states(Nt=trap_2D['Ntrap2D'],
                                                                   Ep=trap_2D['Etrap2D'] - phigrid[ii],
                                                                   Tt=trap_2D['Ttrap2D'], Ef=-Vy, T=self.T,
                                                                   nature=trap_2D['nature']) * (
                                                   2 if ii < self.grid_size - 1 else 1) / (
                                                           self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[
                                                       ii - 1]) \
                                      for trap_2D in self.trap_list_2D])
            Qtrap3D = lambda ii: sum([0 if self.xgrid[ii] <= self.tDE \
                                          else q * gaussian_states(Nt=trap_3D['Ntrap3D'],
                                                                   Ep=trap_3D['Etrap3D'] - phigrid[ii],
                                                                   Tt=trap_3D['Ttrap3D'], Ef=-Vy, T=self.T,
                                                                   nature=trap_3D['nature']) for trap_3D in
                                      self.trap_list_3D])

            F = np.zeros(np.shape(self.xgrid))
            F[0] = Vgs - (self.phiM - self.chiS)
            for ii in range(1, self.grid_size):
                F[ii] = (Qfree(ii) + Qfixed(ii) + Qtrap2D(ii) + Qtrap3D(ii) + Qdop(ii)) * self.sf / epso
            return F

        """
        Solve the poisson equation using scipy.optimize fsolve
        """
        self.phigrid = fsolve(lambda phigrid: Ms @ phigrid - FFsolve(phigrid), self.phigrid)
        error = np.sum(np.abs(Ms @ self.phigrid - FFsolve(self.phigrid)))
        print(f'Final error is {error} Vgs={Vgs}V')

        """
        From the solved phigrid calculate Qfree and Ec
        """
        self.Qfree_grid = np.array(
            [0 if x <= self.tDE else q * self.N3D * FD_int_3D((phi - Vy) / self.phiT) for x, phi in
             zip(self.xgrid, self.phigrid)])
        self.Ec_grid = np.array(
            [self.HfO2_offset - phi if x <= self.tDE else -phi for x, phi in zip(self.xgrid, self.phigrid)])
        self.Qfree = 0
        for ii in range(1, self.grid_size - 1):
            if self.xgrid[ii] > self.tDE:
                self.Qfree += self.Qfree_grid[ii] * (self.xgrid[ii + 1] - self.xgrid[ii - 1]) / 2
        self.Qfree += self.Qfree_grid[-1] * (self.xgrid[-1] - self.xgrid[-2])
        return None

    def schrodinger_solve(self, phigrid):
        """
        Solve schrodinger equation inside
        """
        bias = 0  # Lanczos Bias
        t = hred ** 2 / (2 * q * self.me)  # schrodinger pre-factor
        xgrid = self.xgrid[self.xgrid > self.tDE]
        phigrid = phigrid[self.xgrid > self.tDE]
        Hmat = np.diag(np.zeros(np.shape(phigrid)))

        xgrid = np.insert(xgrid, 0, 2 * xgrid[0] - xgrid[1])
        xgrid = np.insert(xgrid, len(xgrid), 2 * xgrid[-1] - xgrid[-2])
        for ii in range(len(phigrid)):
            jj = ii + 1
            if ii != len(phigrid) - 1:
                Hmat[ii, ii + 1] = -2 * t / ((xgrid[jj + 1] - xgrid[jj]) * (xgrid[jj + 1] - xgrid[jj - 1]))
            Hmat[ii, ii] = (2 / ((xgrid[jj + 1] - xgrid[jj])) + 2 / ((xgrid[jj] - xgrid[jj - 1]))) * t / (
                        xgrid[jj + 1] - xgrid[jj - 1]) - phigrid[ii] + bias
            if ii != 0:
                Hmat[ii, ii - 1] = -2 * t / ((xgrid[jj] - xgrid[jj - 1]) * (xgrid[jj + 1] - xgrid[jj - 1]))

        Hmat = sparse.csr_matrix(Hmat)
        E, V = las.eigsh(Hmat, k=self.num_levels, which='SA')
        E -= bias
        return E, V

    def coupled_poisson_schrodinger_solve(self, Vgs, Vy):
        """
        Initialize phigrid
        """
        self.phigrid[0] = Vgs - (self.phiM - self.chiS)
        self.phigrid[1:] = self.phigrid[1:] * min(0, self.phigrid[0]) + self.phiT

        Ms = sparse.csr_matrix(self.poisson_mat)  # sparse matrix version of poisson_mat
        """
        Forcing function of the poisson equation calculated from the charges
        """
        def FFsolve(phigrid):
            E, V = self.schrodinger_solve(phigrid)
            def Qfree(ii):
                iiDE = np.where(self.xgrid == self.tDE)[0] + 1
                if self.xgrid[ii] <= self.tDE:
                    return 0
                else:
                    qfree = 0
                    for jj in range(len(E)):
                        qfree += q * self.N2D * log(1 + exp((-E[jj] - Vy) / (kB * self.T))) * np.abs(
                            V[ii - iiDE, jj] ** 2) * (2 if ii < self.grid_size - 1 else 1) / (
                                             self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[ii - 1])
                    return qfree

            def Qtail(ii):
                iiDE = np.where(self.xgrid == self.tDE)[0] + 1
                if self.xgrid[ii] <= self.tDE:
                    return 0
                else:
                    qtail = sum([q * exponential_states(Nt=tail_2D['Ntail2D'], Tt=tail_2D['Ttail2D'], Ep=E[0], Ef=-Vy,
                                                        T=self.T, nature=tail_2D['nature']) * np.abs(
                        V[ii - iiDE, 0] ** 2) * (2 if ii < self.grid_size - 1 else 1) / (
                                             self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[ii - 1])
                                 for tail_2D in self.band_tail_list_2D])
                    return qtail

            Qdop = lambda ii: 0 if self.xgrid[ii] <= self.tDE else -1 * q * self.NCH

            Qfixed = lambda ii: sum([0 if self.xgrid[ii] != fixed_2D['xfixed'] \
                                         else q * fixed_2D['Nfixed'] * (2 if ii < self.grid_size - 1 else 1) / (
                        self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[ii - 1]) \
                                     for fixed_2D in self.fixed_charge_list_2D])

            Qtrap2D = lambda ii: sum([0 if self.xgrid[ii] != trap_2D['xtrap2D'] \
                                          else q * gaussian_states(Nt=trap_2D['Ntrap2D'],
                                                                   Ep=trap_2D['Etrap2D'] - phigrid[ii],
                                                                   Tt=trap_2D['Ttrap2D'], Ef=-Vy, T=self.T,
                                                                   nature=trap_2D['nature']) * (
                                                   2 if ii < self.grid_size - 1 else 1) / (
                                                           self.xgrid[min(ii + 1, self.grid_size - 1)] - self.xgrid[
                                                       ii - 1]) \
                                      for trap_2D in self.trap_list_2D])

            Qtrap3D = lambda ii: sum([0 if self.xgrid[ii] <= self.tDE \
                                          else q * gaussian_states(Nt=trap_3D['Ntrap3D'],
                                                                   Ep=trap_3D['Etrap3D'] - phigrid[ii],
                                                                   Tt=trap_3D['Ttrap3D'], Ef=-Vy, T=self.T,
                                                                   nature=trap_3D['nature']) for trap_3D in
                                      self.trap_list_3D])

            F = np.zeros(np.shape(self.xgrid))
            QFREE = np.zeros(np.shape(self.xgrid))
            QTAIL = np.zeros(np.shape(self.xgrid))
            F[0] = Vgs - (self.phiM - self.chiS)
            QFREE[0] = Qfree(0)
            QTAIL[0] = Qtail(0)
            for ii in range(1, self.grid_size):
                QFREE[ii] = Qfree(ii)
                QTAIL[ii] = Qtail(ii)
                F[ii] = (QFREE[ii] + QTAIL[ii] + Qfixed(ii) + Qtrap2D(ii) + Qtrap3D(ii) + Qdop(ii)) * self.sf / epso
            return F, QFREE, QTAIL

        """
        Solve the poisson and schrodinger equations self-consistently
        """
        alpha = 0.05
        tol = 2e-6
        max_num_iterations = 500
        phigrid_old = self.phigrid
        F = np.zeros(np.shape(self.xgrid))
        QFREE = np.zeros(np.shape(self.xgrid))
        QTAIL = np.zeros(np.shape(self.xgrid))
        for iteration in range(max_num_iterations):
            F, QFREE, QTAIL = FFsolve(phigrid_old)
            phigrid_new = las.spsolve(A=Ms, b=F)
            error = (np.sum(np.abs(phigrid_new - phigrid_old))) / len(phigrid_old)
            if error < tol:
                print(f'Error below tolerance. Converged after {iteration} iterations!!')
                print(f'Final error is {error} for Vgs={Vgs}V')
                break
            phigrid_old += alpha * (phigrid_new - phigrid_old)
            if iteration == max_num_iterations - 1:
                print('Warning ! Convergence not reached, even after max. iterations!!')
                print(f'Final error is {error} for Vgs={Vgs}V')
        self.phigrid = phigrid_new

        """
        From the solved phigrid calculate Qfree and Ec
        """
        Eprint, Vprint = self.schrodinger_solve(self.phigrid)
        print(f'Ground State at {Eprint[0] + self.phigrid[-1]}eV above reference level')
        self.Qfree_grid = QFREE
        self.Qtail_grid = QTAIL
        self.Ec_grid = np.array(
            [self.HfO2_offset - phi if x <= self.tDE else -phi for x, phi in zip(self.xgrid, self.phigrid)])
        self.Qfree = 0
        self.Qtail = 0
        for ii in range(1, self.grid_size - 1):
            if self.xgrid[ii] > self.tDE:
                self.Qfree += self.Qfree_grid[ii] * (self.xgrid[ii + 1] - self.xgrid[ii - 1]) / 2
                self.Qtail += self.Qtail_grid[ii] * (self.xgrid[ii + 1] - self.xgrid[ii - 1]) / 2
        self.Qfree += self.Qfree_grid[-1] * (self.xgrid[-1] - self.xgrid[-2])
        self.Qtail += self.Qtail_grid[-1] * (self.xgrid[-1] - self.xgrid[-2])
        return None

    def band_plotter(self, Vgs, Vy, mode='sp'):
        """
        Plot the band diagram for a given Vgs and Vy
        """
        if mode == 'sp':
            self.coupled_poisson_schrodinger_solve(Vgs=Vgs, Vy=Vy)
        else:
            self.poisson_solve(Vgs=Vgs, Vy=Vy)
        lin_plot(x=[self.xgrid], y=[self.Ec_grid], c=['r'], s=['solid'], figname='./band_structure_profile.png')
        return None

    def Id_full(self, Vgs, Vds, mode='sp'):
        """
        Solve for different y and integrate along the channel direction under GCA to calculate Id
        """
        Vy_all = np.linspace(0, Vds, 33)
        Iy_all = np.zeros(np.shape(Vy_all))
        Qch = 0
        for i, Vy in enumerate(Vy_all):
            if mode == 'sp':
                self.coupled_poisson_schrodinger_solve(Vgs=Vgs, Vy=Vy)
            else:
                self.poisson_solve(Vgs=Vgs, Vy=Vy)
            Iy_all[i] = self.mu_band * (self.W / self.L) * self.Qfree
            if Vy == 0:
                # Qch = self.Qfree + self.Qtail
                Qf = self.Qfree
                Qt = self.Qtail
        Id = integrate.simpson(y=Iy_all, x=Vy_all)
        # return Id, Qch
        return Id, Qf, Qt


def Id_full_sweep_parallel_v0(Vgs_sweep, fet_name, foo):
    """
    Parallelize Id calculation for a given Vgs sweep
    """
    t1 = time.time()
    outputs = multiple_workers(foo=foo, inputs=Vgs_sweep)
    t2 = time.time()
    print('Time taken is ', t2 - t1, 's')
    # Id_sweep = np.array([id for id, qch in outputs])
    # Qch_sweep = np.array([qch for id, qch in outputs])
    # return Qch_sweep, Id_sweep, Vgs_sweep
    Id_sweep = np.array([id for id, qf, qt in outputs])
    Qf_sweep = np.array([qf for id, qf, qt in outputs])
    Qt_sweep = np.array([qt for id, qf, qt in outputs])
    return Qf_sweep, Qt_sweep, Id_sweep, Vgs_sweep

def IdVd_full_sweep_parallel_v0(Vds_sweep, fet_name, foo):
    """
    Parallelize Id calculation for a given Vds sweep
    """
    t1 = time.time()
    outputs = multiple_workers(foo=foo, inputs=Vds_sweep)
    t2 = time.time()
    print('Time taken is ', t2 - t1, 's')
    # Id_sweep = np.array([id for id, qch in outputs])
    # Qch_sweep = np.array([qch for id, qch in outputs])
    # return Qch_sweep, Id_sweep, Vds_sweep
    Id_sweep = np.array([id for id, qf, qt in outputs])
    Qf_sweep = np.array([qf for id, qf, qt in outputs])
    Qt_sweep = np.array([qt for id, qf, qt in outputs])
    return Qf_sweep, Qt_sweep, Id_sweep, Vds_sweep

"""
numerical model fit for sispad 2024 talk
"""
if __name__ == "__main__":
    def parse_value(value):
        try:
            return ast.literal_eval(value)
        except:
            return value.strip()

    num_params = {}
    # Define the directory to start the search (current working directory)
    start_directory = os.getcwd()
    with open(os.path.join(start_directory, "sispad_talk_baseline", "vlsi25_num_model_sispad_params.txt"), 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':', 1)
                num_params[key.strip()] = parse_value(value)

    print(num_params)

    np_d = np.loadtxt(os.path.join(start_directory, "sispad_talk_baseline", "die22_2t1_w12l2_idvg_vd50m_2V_3rd.csv"), delimiter=',')

    Lg, W = num_params['L'], num_params['W']
    tDE, kDE = num_params["tDE"], num_params["kDE"]
    Cg = epso * kDE / tDE

    Vgs_all_mod = []
    Vds_all_mod = []
    Qch_all_mod = []
    Qf_all_mod = []
    Qt_all_mod = []
    Id_all_mod = []

    Vds = [50e-3, 2]
    for ii in range(len(Vds)):
        num_params['Vds'] = Vds[ii]
        osfet_dut_0 = OSFET_numerical_model_v1_0_0(params=num_params)


        def foo_gen(Vgs, Vds, fet_dut):
            print(f'Parallel Solving for Vgs={Vgs}V and tCH={fet_dut.tCH}m')
            return fet_dut.Id_full(Vgs, Vds, mode='sp')


        def foo_dut_0(Vgs):
            return foo_gen(Vgs=Vgs, Vds=num_params['Vds'], fet_dut=osfet_dut_0)


        N1 = len(np_d[:, 0])
        Vgs_sweep = np_d[:int(N1 / 2):1, 0]

        # Qch_num_sweep, Id_num_sweep, Vgs_sweep = Id_full_sweep_parallel_v0(Vgs_sweep=Vgs_sweep, fet_name=osfet_dut_0, foo=foo_dut_0)
        Qf_num_sweep, Qt_num_sweep, Id_num_sweep, Vgs_sweep = Id_full_sweep_parallel_v0(Vgs_sweep=Vgs_sweep, fet_name=osfet_dut_0, foo=foo_dut_0)
        Qch_num_sweep = Qf_num_sweep + Qt_num_sweep

        Vgs_all_mod += [np_d[:int(N1 / 2):1, 0]]
        Id_all_mod += [abs(np_d[:int(N1 / 2):1, ii + 1] * 1e6 / (W * 1e6))]
        Vds_all_mod += [num_params['Vds']]

        Vgs_all_mod += [Vgs_sweep]
        Id_all_mod += [Id_num_sweep * 1e6 / (W * 1e6)]
        Vds_all_mod += [num_params['Vds']]
        Qch_all_mod += [Qch_num_sweep * Lg * W]
        Qf_all_mod += [Qf_num_sweep * Lg * W]
        Qt_all_mod += [Qt_num_sweep * Lg * W]

    plt.rcParams["figure.figsize"] = (13, 10)
    legend = [None]
    logy_lin_plot(x=Vgs_all_mod, y=Id_all_mod, c=['k', 'r'], lw=4, s=['None', 'solid'], m=['o', 'None'], xlabel="$V_{GS} (V)$", ylabel="$Id (\mu A/\mu m)$", l=legend, ylim=[1e-8, 1e1], figname='Id.png')
    logy_lin_plot(x=Vgs_all_mod[1:2] * 2, y=Qch_all_mod[:1] + Qf_all_mod[:1], c=['b', 'b'], lw=4, s=['solid', 'dashed'], m=['None', 'None'], xlabel="$V_{GS} (V)$", ylabel="$Qch (C)$", l=legend, figname='Qch.png')
