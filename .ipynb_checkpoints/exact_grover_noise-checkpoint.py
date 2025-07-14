#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pennylane as q


# In[68]:


def oracle(combo, phi, p, wires):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=wires[0]+i)
            q.DepolarizingChannel(p, wires=wires[0]+i)
    q.ctrl(q.PhaseShift, control=wires[:-1])(phi, wires=[wires[-1]])
    for wire in wires:
        q.DepolarizingChannel(p, wires=wire)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=wires[0]+i)
            q.DepolarizingChannel(p, wires=wires[0]+i)

def hadamard_transform(p, wires):
    for wire in wires:
        q.Hadamard(wires=wire)
        q.DepolarizingChannel(p, wires=wire)

def diffusion(n_bits, phi, p, wires):
    hadamard_transform(p, wires)
    for wire in wires:
        q.PauliX(wires=wire)
        q.DepolarizingChannel(p, wires=wire)
    q.ctrl(q.PhaseShift, control=wires[:-1])(phi, wires=[wires[-1]])
    for wire in wires:
        q.DepolarizingChannel(p, wires=wire)
    for wire in wires:
        q.PauliX(wires=wire)
        q.DepolarizingChannel(p, wires=wire)
    hadamard_transform(p, wires)

def modified_grover_iter(combo, p, wires):
    n_bits = len(combo)
    theta=np.arcsin(np.sqrt(1/(2**n_bits)))
    J=np.floor((np.pi/2-theta)/(2*theta))
    phi=np.round(2*np.arcsin(np.sin(np.pi/(4*J+6))/np.sin(theta)), 4)
    
    hadamard_transform(p, wires)
    for _ in range(J.astype(int)+1):
        oracle(combo, phi, p, wires)
        diffusion(n_bits, phi, p, wires)

