#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pennylane as q


# In[5]:


def oracle(combo, p):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
            q.DepolarizingChannel(p, wires=i)
    q.ctrl(q.PauliZ, control=list(range(n_bits-1)))(wires=[n_bits-1])
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

def diffusion(n_bits, all_wires, p):
    hadamard_transform(all_wires, p)
    for i in range(n_bits):
        q.PauliX(wires=i)
        q.DepolarizingChannel(p, wires=i)
    q.ctrl(q.PauliZ, control=list(range(n_bits-1)))(wires=[n_bits-1])
    for i in range(n_bits):
        q.DepolarizingChannel(p, wires=i)
    for i in range(n_bits):
        q.PauliX(wires=i)
        q.DepolarizingChannel(p, wires=i)
    hadamard_transform(all_wires, p)

def original_grover_iter(combo, p):
    n_bits = len(combo)
    G_steps=np.floor((np.pi/4)*(np.sqrt(2**len(combo)))).astype(int)
    all_wires = list(range(n_bits))
    dev = q.device("default.mixed", wires=n_bits, shots=1000, seed=42)

    @q.qnode(dev)
    def inner_circuit():
        hadamard_transform(all_wires, p)
        for _ in range(G_steps):
            oracle(combo, p)
            diffusion(n_bits, all_wires, p)
        return q.probs(wires=all_wires)

    return inner_circuit()

