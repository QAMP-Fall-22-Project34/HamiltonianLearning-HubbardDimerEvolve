from math import pi
import numpy as np
import retworkx as rx
import scipy.linalg as sl
from qiskit_nature.problems.second_quantization.lattice import (
    BoundaryCondition,
    Lattice,
    LatticeDrawStyle,
    LineLattice, 
    FermiHubbardModel
)
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.opflow import StateFn, OperatorStateFn, PrimitiveOp, PauliExpectation, CircuitStateFn, PauliTrotterEvolution, CircuitSampler, MatrixEvolution, Suzuki, X, Y, Z, I, Plus, Minus, Zero, One
from typing import List


# Define some Type Aliases
Vector = List[float]

class DimerLattice:
  # instance attributes
  def __init__(self, num_particles: float, hopping_parameter: float, onsite_interaction: float, vext: Vector ) -> None:
    self.nsites = 2
    self.n_fermionic_modes = 2*self.nsites
    self.num_particles = num_particles
    self.t = hopping_parameter
    self.U = onsite_interaction
    self.vext = vext
    self.jw_converter = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
    self.setupOps()
    self.setupQubitOps()

  # instance method
  def setupOps(self):
    self.Hop   = self.getFHMOp(self.t, self.U, self.vext).reduce()
    self.Top   = self.getFHMOp(self.t, 0, np.zeros(len(self.vext))).reduce()
    self.TpUop = self.getFHMOp(self.t, self.U, np.zeros(len(self.vext))).reduce()
    self.Uop   = self.getFHMOp(0, self.U, np.zeros(len(self.vext))).reduce()
    self.Vop   = self.getFHMOp(0, 0, self.vext).reduce()
    self.mfHop = self.getFHMOp(self.t, 0, self.vext).reduce()
    #Set up number operator for Hubbard dimer
    self.N_op = sum(FermionicOp("+_"+str(i)+" -_"+str(i), register_length=self.n_fermionic_modes, display_format='dense') for i in range(2*self.nsites))
    #Set up spin-resolved site occupation
    self.nsigma = [ FermionicOp("+_"+str(i)+" -_"+str(i), register_length=self.n_fermionic_modes, display_format='dense') for i in range(2*self.nsites) ]
    #Set up total-site occupation op
    self.nsite = [ FermionicOp("+_"+str(2*i)+" -_"+str(2*i), register_length=self.n_fermionic_modes, display_format='dense') + 
                      FermionicOp("+_"+str(2*i+1)+" -_"+str(2*i+1), register_length=self.n_fermionic_modes, display_format='dense') for i in range(self.nsites) ]
    #Set up the spin-density operator
    self.mz = [ FermionicOp("+_"+str(2*i)+" -_"+str(2*i), register_length=self.n_fermionic_modes, display_format='dense') + 
                  (-1.0)*FermionicOp("+_"+str(2*i+1)+" -_"+str(2*i+1), register_length=self.n_fermionic_modes, display_format='dense') for i in range(self.nsites) ]

  #instance method
  def setupQubitOps(self):
    self.qH   = self.jw_converter.convert(self.Hop, num_particles=self.num_particles)
    self.qT   = self.jw_converter.convert(self.Top, num_particles=self.num_particles)
    self.qTpU = self.jw_converter.convert(self.TpUop, num_particles=self.num_particles)
    self.qU   = self.jw_converter.convert(self.Uop, num_particles=self.num_particles)
    self.qV   = self.jw_converter.convert(self.Vop, num_particles=self.num_particles)
    self.qmfH = self.jw_converter.convert(self.mfHop, num_particles=self.num_particles)
    self.qN   = self.jw_converter.convert(self.N_op, num_particles=self.num_particles)
    self.qnsigma   = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.nsigma ]
    self.qnsite = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.nsite ]
    self.qmz  = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.mz ]

  #instance method
  def densityPenaltyOp(self, ni: Vector):
    #Construct the operator list [ n_i_op - ni ]
    dp_op = [ self.nsite[i] + (-1.0 * ni[i])*FermionicOp.one(self.n_fermionic_modes) for i in range(len(ni) - 1) ]
    qdp_op = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in dp_op ]
    return qdp_op 

  # instance method
  def getFHMOp(self, t: float, U: float, vext: Vector):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(self.nsites))
    weighted_edge_list = [
        (0, 0, vext[0]),
        (1, 1, vext[1]),
        (0, 1, t),
    ]
    graph.add_edges_from(weighted_edge_list)
    general_lattice = Lattice(graph)
    set(general_lattice.graph.weighted_edge_list())
    fhm = FermiHubbardModel(general_lattice, onsite_interaction=U)
    return fhm.second_q_ops()

class PeierlsDimerLattice(DimerLattice):
  def __init__(self, num_particles: float, hopping_parameter: float, onsite_interaction: float, vext: Vector, vpA: float) -> None:
    self.nsites = 2
    self.n_fermionic_modes = 2*self.nsites
    self.num_particles = num_particles
    self.vPA = vpA
    self.hopping_parameter = hopping_parameter
    self.t = hopping_parameter*np.exp(-(1j)*vpA)
    self.U = onsite_interaction
    self.vext = vext
    self.jw_converter = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
    self.setupOps()
    self.setupQubitOps()

  # instance method
  def getFHMOp(self, t: complex, U: float, vext: Vector):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(self.nsites))
    weighted_edge_list = [
        (0, 0, vext[0]),
        (1, 1, vext[1]),
        (0, 1, t),
    ]
    graph.add_edges_from(weighted_edge_list)
    general_lattice = Lattice(graph)
    set(general_lattice.graph.weighted_edge_list())
    fhm = FermiHubbardModel(general_lattice, onsite_interaction=U)
    return fhm.second_q_ops()


class PeierlsHolsteinDimerLattice(PeierlsDimerLattice):
  def __init__(self, num_particles: float, hopping_parameter: float, onsite_interaction: float, vext: Vector, vpA: float, n_ph: int, nq_ph: int, ph_freq: Vector, el_ph_g: Vector) -> None:
    self.nsites = 2
    self.n_fermionic_modes = 2*self.nsites
    self.n_ph_modes = n_ph # number of ph modes
    self.q_per_ph = nq_ph # number of qubits per ph mode
    self.q_ph_tot = n_ph * nq_ph #Total number of qubits to rep all phonon modes
    assert len(ph_freq) == n_ph
    self.ph_freq = ph_freq
    assert len(el_ph_g) == n_ph
    self.el_ph_g = el_ph_g
    self.num_particles = num_particles
    self.vPA = vpA
    self.hopping_parameter = hopping_parameter
    self.t = hopping_parameter*np.exp(-(1j)*vpA)
    self.U = onsite_interaction
    self.vext = vext
    self.jw_converter = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
    self.setupOps()
    self.setupQubitOps()
    self.tot_qubits = self.qH.num_qubits + self.q_ph_tot
    self.setup_ph_ops()
    self.setup_eldip()
    self.setup_elph_ops()

  def setup_ph_ops(self):
        #Currently hard coded for n_ph = 1 and nq_ph = 2
        assert self.n_ph_modes == 1
        assert self.q_per_ph == 2
        half = 0.5
        # This is the diagonal phonon mode energy ( a^dag a + 1/2 ) * h_bar * w
        self.q_ph_H = (1./4) * ( (I^I) + (I^Z) + (Z^I) + (Z^Z) ) * half * self.ph_freq[0] + \
                      (1./4) * ( (I^I) - (I^Z) + (Z^I) - (Z^Z) ) * (half + 1) * self.ph_freq[0] + \
                      (1./4) * ( (I^I) + (I^Z) - (Z^I) - (Z^Z) ) * (half + 2) * self.ph_freq[0] + \
                      (1./4) * ( (I^I) - (I^Z) - (Z^I) + (Z^Z) ) * (half + 3) * self.ph_freq[0] 
        self.q_ph_N = (1./4) * ( (I^I) - (I^Z) + (Z^I) - (Z^Z) ) * (1) + \
                      (1./4) * ( (I^I) + (I^Z) - (Z^I) - (Z^Z) ) * (2) + \
                      (1./4) * ( (I^I) - (I^Z) - (Z^I) + (Z^Z) ) * (3)
        # setup the a^dag + a operator
        self.q_ph_adpa = ((np.sqrt(3) + 1)/2) * (I^X) + ((1 - np.sqrt(3))/2) * (Z^X) + (1/np.sqrt(2)) * (X^X) + (1/np.sqrt(2)) * (Y^Y) 
        
  def setup_eldip(self):
      #Set up number operator for Hubbard dimer
      self.dip_op = FermionicOp("+_"+str(0)+" -_"+str(0), register_length=self.n_fermionic_modes, display_format='dense') + \
                    FermionicOp("+_"+str(1)+" -_"+str(1), register_length=self.n_fermionic_modes, display_format='dense') - \
                    FermionicOp("+_"+str(2)+" -_"+str(2), register_length=self.n_fermionic_modes, display_format='dense') - \
                    FermionicOp("+_"+str(3)+" -_"+str(3), register_length=self.n_fermionic_modes, display_format='dense')
      self.q_dip  = self.jw_converter.convert(self.dip_op, num_particles=self.num_particles)

  def setup_elph_ops(self):
      self.q_el_ham = (I^I) ^ self.qH
      self.q_ph_ham = self.q_ph_H ^ (I^I^I^I)
      self.q_elph = (-1.0) * self.el_ph_g[0] * ( self.q_ph_adpa ^ self.q_dip )
      self.q_elph_Ham = self.q_el_ham + self.q_ph_ham + self.q_elph
      self.q_el_N = (I^I)^self.qN
      self.q_ph_occ = self.q_ph_N ^ (I^I^I^I) 
      self.q_el_dip = (I^I)^self.q_dip