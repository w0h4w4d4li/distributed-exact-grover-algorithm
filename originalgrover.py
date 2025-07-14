#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pennylane as q


# In[138]:


def oracle(combo):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
    q.ctrl(q.PauliZ, control=list(range(n_bits-1)))(wires=[n_bits-1])
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)

def hadamard_transform(all_wires):
    for wire in all_wires:
        q.Hadamard(wires=wire)

def diffusion(n_bits, all_wires):
    hadamard_transform(all_wires)
    for i in range(n_bits):
        q.PauliX(wires=i)
    q.ctrl(q.PauliZ, control=list(range(n_bits-1)))(wires=[n_bits-1])
    for i in range(n_bits):
        q.PauliX(wires=i)
    hadamard_transform(all_wires)

def original_grover_iter(combo):
    n_bits = len(combo)
    G_steps=np.floor((np.pi/4)*(np.sqrt(2**len(combo)))).astype(int)
    all_wires = list(range(n_bits))
    dev = q.device("default.qubit", wires=n_bits, shots=1000)

    @q.qnode(dev)
    def inner_circuit():
        hadamard_transform(all_wires)
        for _ in range(G_steps):
            oracle(combo)
            diffusion(n_bits, all_wires)
        return q.probs(wires=all_wires)

    return inner_circuit()


# In[144]:


combo=[1,0,0,1]
original_grover_iter(combo)

