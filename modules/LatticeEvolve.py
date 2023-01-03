from Lattices import PeierlsDimerLattice, Vector
from LatticeSolver import LatticeSolver
import numpy as np
import scipy.linalg as sl 

class PeierlsDimerEvolve(PeierlsDimerLattice):
    
    def __init__(self, num_particles: float, hopping_parameter: float, onsite_interaction: float, vext: Vector, vpA: float) -> None:
        super().__init__(num_particles, hopping_parameter, onsite_interaction, vext, vpA)
        self.wfn_start = None
        self.wfn_evolved = None
        self.eig_spectrum = None
        self.eig_states = None
        self.num_sector_states = None
        self.ndim = None

    def update_ops(self, vpA: float):
        self.vPA = vpA
        self.t = self.hopping_parameter*np.exp(-(1j)*vpA)
        self.setupOps()
        self.setupQubitOps()

    def solve_for_spectrum(self):
        latSolver = LatticeSolver(self)
        result = latSolver.classical_solve([self.qH, self.qN])
        idx_e_N = np.array([result[0][i] for i in range(len(result[0]))])
        self.eig_spectrum = idx_e_N[:,1]
        self.eig_states = result[1]
        self.num_sector_states = len(self.eig_spectrum)
        self.ndim = len(self.eig_states[0])
        print(self.eig_spectrum, self.num_sector_states, self.ndim)
        return self.eig_spectrum

    def set_starting_state(self,coeffs):
        assert len(coeffs) == self.num_sector_states
        coeffs = coeffs/np.linalg.norm(coeffs)
        self.wfn_start = np.zeros(self.ndim, dtype=complex)
        for i in range(len(coeffs)):
            self.wfn_start += coeffs[i] * self.eig_states[i]
        print(repr(self.wfn_start))
        #calculate overlap with eig_states
        for i in range(self.num_sector_states):
            ovlp_i = np.dot(self.wfn_start.conj().T,self.eig_states[i])
            print(f"overlap with state {i} is {ovlp_i}")
        #compute energy expectation of starting state:
        self.Hmat = self.qH.to_matrix()
        e_start = np.dot(self.wfn_start.conj().T,np.dot(self.Hmat, self.wfn_start))
        print(f'stating energy is {e_start}')
        self.wfn_evolved = self.wfn_start
    
    def time_evolve(self,T):
        Umat  = sl.expm(-(1j * T)*self.Hmat)
        self.wfn_evolved = np.dot(Umat, self.wfn_start)
        #calculate density matrix in eigen basis \rho_ij = < i| psi > <psi| j>

    def get_dm_element(self, ri, rj):
        ovlp_i = np.dot(self.eig_states[ri].conj().T,self.wfn_evolved)
        ovlp_j = np.dot(self.wfn_evolved.conj().T,self.eig_states[rj])
        # print(f"rho({ri},{rj}) = {ovlp_i * ovlp_j}")    
        return ovlp_i * ovlp_j

    def get_energy(self):
        #compute energy expectation of evolved state:
        return np.dot(self.wfn_evolved.conj().T,np.dot(self.Hmat, self.wfn_evolved))

    def get_site_density(self):
        rho_t = np.zeros(self.nsites)
        for iop in range(len(self.qnsite)):
            opm = self.qnsite[iop].to_matrix()
            rho_t[iop] = np.real(np.dot(self.wfn_evolved.conj().T,np.dot(opm,self.wfn_evolved)))
        return np.real(rho_t)

    def get_site_rdm(self):
        rho = np.outer(self.wfn_evolved.conj(), self.wfn_evolved)
        rho_tensor = rho.reshape((4,4,4,4))
        return np.trace(rho_tensor, axis1=0, axis2=2)

    def get_spinorb_rdm(self):
        rho = np.outer(self.wfn_evolved.conj(), self.wfn_evolved)
        rho_tensor = rho.reshape((8,2,8,2))
        return np.trace(rho_tensor, axis1=0, axis2=2)        
        

    def trotter_evolve(self,tt,At):
        nt = len(tt)
        assert len(tt) == len(At)
        dt = tt[1]-tt[0]
        ene_t = np.zeros(nt)
        rho_t = []
        left_rdm = []
        spo_rdm = []
        for i in range(nt):
            #re-build the time-dependent hamiltonian
            self.update_ops(At[i])
            self.Hmat = self.qH.to_matrix()
            Umat  = sl.expm(-(1j * dt)*self.Hmat)
            self.wfn_evolved = np.dot(Umat, self.wfn_evolved)
            left_rdm.append(self.get_site_rdm())
            spo_rdm.append(self.get_spinorb_rdm())
            ene_t[i] = np.real(self.get_energy())
            rho_t.append(self.get_site_density())
        return (np.array(spo_rdm, dtype=complex), np.array(left_rdm, dtype=object), np.array(rho_t), ene_t)
