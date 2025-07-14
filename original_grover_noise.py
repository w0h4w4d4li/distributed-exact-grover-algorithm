#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pennylane as q


# In[72]:


def oracle(combo, p, wires):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=wires[0]+i)
            q.DepolarizingChannel(p, wires=wires[0]+i)
    q.ctrl(q.PauliZ, control=wires[:-1])(wires=[wires[-1]])
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

def diffusion(n_bits, p, wires):
    hadamard_transform(p, wires)
    for wire in wires:
        q.PauliX(wires=wire)
        q.DepolarizingChannel(p, wires=wire)
    q.ctrl(q.PauliZ, control=wires[:-1])(wires=[wires[-1]])
    for wire in wires:
        q.DepolarizingChannel(p, wires=wire)
    for wire in wires:
        q.PauliX(wires=wire)
        q.DepolarizingChannel(p, wires=wire)
    hadamard_transform(p, wires)

def original_grover_iter(combo, p, wires):
    n_bits = len(combo)
    G_steps=np.floor((np.pi/4)*(np.sqrt(2**len(combo)))).astype(int)

    hadamard_transform(p, wires)
    for _ in range(G_steps):
        oracle(combo, p, wires)
        diffusion(n_bits, p, wires)

