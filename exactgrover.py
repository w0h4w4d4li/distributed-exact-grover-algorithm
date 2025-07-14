#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pennylane as q


# In[47]:


def oracle(combo, phi):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
    q.ctrl(q.PhaseShift, control=list(range(n_bits-1)))(phi, wires=[n_bits-1])
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
    q.ctrl(q.PhaseShift, control=list(range(n_bits-1)))(phi, wires=[n_bits-1])
    for i in range(n_bits):
        q.PauliX(wires=i)
    hadamard_transform(all_wires)

def modified_grover_iter(combo):
    n_bits = len(combo)
    
    theta=np.arcsin(np.sqrt(1/(2**n_bits)))
    J=np.floor((np.pi/2-theta)/(2*theta))
    phi=np.round(2*np.arcsin(np.sin(np.pi/(4*J+6))/np.sin(theta)), 5)
    
    all_wires = list(range(n_bits))
    dev = q.device("default.qubit", wires=n_bits, shots=1)

    @q.qnode(dev)
    def inner_circuit():
        hadamard_transform(all_wires)
        for _ in range(J.astype(int)+1):
            oracle(combo, phi)
            diffusion(n_bits, all_wires, phi)
        return q.probs(wires=all_wires)

    return inner_circuit()


# In[79]:


combo=[0,0,1,0]
modified_grover_iter(combo)

