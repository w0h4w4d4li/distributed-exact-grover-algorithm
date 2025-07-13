#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pennylane as q


# In[89]:


def oracle(combo):
    n_bits=len(combo)
    for i, bit in enumerate(combo):
        if bit == 0:
            q.PauliX(wires=i)
    q.ctrl(q.PauliX, control=list(range(n_bits)))(wires=[n_bits])
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
    q.ctrl(q.PauliX, control=list(range(n_bits)))(wires=[n_bits])
    for i in range(n_bits):
        q.PauliX(wires=i)
    hadamard_transform(all_wires)

def original_grover_iter(combo, num_steps):
    n_bits = len(combo)
    query_register = list(range(n_bits))
    aux = [n_bits]
    all_wires = query_register + aux
    dev = q.device("default.qubit", wires=all_wires, shots=10000)

    @q.qnode(dev)
    def inner_circuit():
        q.PauliX(wires=aux)
        hadamard_transform(all_wires)
        for _ in range(num_steps):
            oracle(combo)
            diffusion(n_bits, all_wires)
        return q.probs(wires=query_register)

    return inner_circuit()


# In[94]:


combo=[1,0,0]
original_grover_iter(combo, np.ceil((np.pi/4)*(np.sqrt(2**len(combo)))).astype(int))

