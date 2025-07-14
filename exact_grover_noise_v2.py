#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pennylane as q


# In[24]:


def oracle(combo, phi, p):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
            q.DepolarizingChannel(p, wires=i)
    q.ctrl(q.PhaseShift, control=list(range(n_bits-1)))(phi, wires=[n_bits-1])
    for i in range(n_bits):
        q.DepolarizingChannel(p, wires=i)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
            q.DepolarizingChannel(p, wires=i)

def hadamard_transform(all_wires, p):
    for wire in all_wires:
        q.Hadamard(wires=wire)
        q.DepolarizingChannel(p, wires=wire)

def diffusion(n_bits, all_wires, phi, p):
    hadamard_transform(all_wires, p)
    for i in range(n_bits):
        q.PauliX(wires=i)
        q.DepolarizingChannel(p, wires=i)
    q.ctrl(q.PhaseShift, control=list(range(n_bits-1)))(phi, wires=[n_bits-1])
    for i in range(n_bits):
        q.DepolarizingChannel(p, wires=i)
    for i in range(n_bits):
        q.PauliX(wires=i)
        q.DepolarizingChannel(p, wires=i)
    hadamard_transform(all_wires, p)

def modified_grover_iter(combo, p):
    n_bits = len(combo)
    
    theta=np.arcsin(np.sqrt(1/(2**n_bits)))
    J=np.floor((np.pi/2-theta)/(2*theta))
    phi=np.round(2*np.arcsin(np.sin(np.pi/(4*J+6))/np.sin(theta)), 5)
    
    all_wires = list(range(n_bits))
    dev = q.device("default.mixed", wires=n_bits, shots=1000, seed=42)

    @q.qnode(dev)
    def inner_circuit():
        hadamard_transform(all_wires, p)
        for _ in range(J.astype(int)+1):
            oracle(combo, phi, p)
            diffusion(n_bits, all_wires, phi, p)
        return q.probs(wires=all_wires)

    return inner_circuit()

