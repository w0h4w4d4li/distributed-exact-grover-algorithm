#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pennylane as q


# In[23]:


def oracle(combo, phi):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
    q.ctrl(q.PhaseShift, control=list(range(n_bits)))(phi, wires=[n_bits])
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)

def hadamard_transform(all_wires):
    for wire in all_wires:
        q.Hadamard(wires=wire)

def diffusion(n_bits, all_wires, phi):
    hadamard_transform(all_wires)
    for i in range(n_bits):
        q.PauliX(wires=i)
    q.ctrl(q.PhaseShift, control=list(range(n_bits)))(phi, wires=[n_bits])
    for i in range(n_bits):
        q.PauliX(wires=i)
    hadamard_transform(all_wires)

def modified_grover_iter(combo, num_steps):
    n_bits = len(combo)
    
    theta=np.arcsin(np.sqrt(1/(2**n_bits)))
    J=np.floor((np.pi/2-theta)/(2*theta))
    phi=np.round(2*np.arcsin(np.sin(np.pi/(4*J+6))/np.sin(theta)), 5)
    
    query_register = list(range(n_bits))
    aux = [n_bits]
    all_wires = query_register + aux
    dev = q.device("default.qubit", wires=all_wires, shots=1)

    @q.qnode(dev)
    def inner_circuit():
        q.PauliX(wires=aux)
        hadamard_transform(all_wires)
        for _ in range(num_steps):
            oracle(combo, phi)
            diffusion(n_bits, all_wires, phi)
        return q.probs(wires=query_register)

    return inner_circuit()


# In[33]:


combo=[0,0,0,0]
theta=np.arcsin(np.sqrt(1/(2**len(combo))))
J=np.floor((np.pi/2-theta)/(2*theta))
modified_grover_iter(combo, J.astype(int)+1)

