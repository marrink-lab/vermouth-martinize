#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:07:14 2017

@author: peterkroon
"""

from martinize2 import *
import networkx as nx
import numpy as np
import scipy.sparse.csgraph as ssc
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import scipy.linalg as sl
import scipy.spatial.distance as ssd
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


PATH = '../molecules/glkfk.pdb'
mol = read_pdb(PATH)
#mol = read_pdb('../molecules/octane.pdb')


def make_laplacian(adjacency_matrix):
#    return ssc.laplacian(adjacency_matrix, normed=True)
    A = ss.csr_matrix(adjacency_matrix, dtype=np.float64)
    D = ss.diags(np.sum(adjacency_matrix, axis=0), dtype=np.float64, format='csc')
    D_inv_sqrt = np.sqrt(ssl.inv(D))
    # Using an unweighted Laplacian does RatioCut (equal degree/size)
    # using a weighted Laplacian does NCut (equal volume, maximize intracluster similarity)
    L = D - A
    # D^(-1/2) * L * D^(-1/2); but then matrix multiplication
    # This works with the * operator, because they're sparse matrices.
    L = D_inv_sqrt * L * D_inv_sqrt
    assert np.allclose(L.A, L.T.A), 'Laplacian is not symmetric'
    return L


def spectral(G, ndim=3):
#    conmat = nx.adjacency_matrix(G, weight='weight')
#    L = make_laplacian(conmat.A)
#    print(L.A)
    L = nx.normalized_laplacian_matrix(G)
    vals, vecs = ssl.eigsh(L, k=ndim+1, sigma=-1e-7, which='LM')
#    S = np.linalg.eig(L.A)
#    order = np.argsort(S[0])
#    S = np.real(S[0]), np.real(S[1])
#    vals = S[0][order]
#    vecs = S[1][:, order]
    scale = 1/np.sqrt(vals[1:ndim+1])
    return vecs[:, 1:ndim+1] * scale

def MDS(G, fill=10, ndim=3):
    n = len(G)
    distmat = nx.adjacency_matrix(G).A
    distmat[distmat == 0] = fill
#    distmat = ssd.squareform(ssd.pdist([G.node[idx]['position'] for idx in G.node]))
    D2 = distmat ** 2
    J = np.eye(n) - 1/n * np.ones(n) * np.ones(n).T
    # Must be matmul
    B = - 0.5 * J @ D2 @ J
    S = np.linalg.eig(B)
    order = np.argsort(S[0])[::-1]
    vals = S[0][order]
    vecs = S[1][:, order]
    M = np.real(vecs[:, :ndim])
    L = np.sqrt(np.abs(vals[:ndim]))
    return M * L

def smacof_dense(G, eps=1e-2, maxiter=5000, ndim=3, start=None):
    n = len(G)
    if start is not None:
        X = np.random.rand(n, ndim)
    else:
        X = start
    disparities = nx.adjacency_matrix(G, weight='distance').A
    w = np.zeros_like(disparities)
    w[disparities != 0] = 1
    w *= (n*(n-1))/(w * disparities**2).sum()
    V = -w
    V[np.arange(len(V)), np.arange(len(V))] = np.sum(w, axis=1)
    V = np.linalg.pinv(V)
    for it in range(maxiter):
        dis = ssd.squareform(ssd.pdist(X))
        dis[dis == 0] = np.inf
        # Update X using the Guttman transform
        S = 1/dis
        B = -w * disparities * S
        B[np.arange(len(B)), np.arange(len(B))] = np.sum(w * disparities * S, axis=1)
        displacement = V @ B
        X = displacement @ X
        # Maybe this should depend on the number of atoms or something
        if np.max(np.abs(np.diag(displacement) - 1)) < eps:
            print('Converged after {} iterations'.format(it))
            break

    return X


def smacof(G, eps=1e-2, maxiter=5000, ndim=3, start=None):
    n = len(G)
    if start is not None:
        X = np.random.rand(n, ndim)
    else:
        X = start
    disparities = nx.adjacency_matrix(G, weight='distance')
    data = [n*(n-1)/disparities.power(2).sum()] * disparities.nnz
    w = ss.csc_matrix((data, disparities.nonzero()), shape=disparities.shape)
    
    # The pseudo-inverse will be dense anyway.
    V = -w.A
    V[np.diag_indices_from(V)] = np.array(w.sum(axis=0))[0]
    V = np.linalg.pinv(V)
    for it in range(maxiter):
        vecs = (X[w.nonzero()[0]] - X[w.nonzero()[1]])
        data = 1/np.sqrt(np.sum(vecs**2, axis=-1))
        S = ss.csr_matrix((data, w.nonzero()), w.shape)
        # Update X using the Guttman transform
        B = -w.multiply(disparities).multiply(S)
        diag = np.array(-B.sum(axis=0))[0]
        B.setdiag(diag)
        # V is dense, so displacements is dense.
        displacement = V @ B
        X = displacement @ X
        # Maybe this should depend on the number of atoms or something
        if np.max(np.abs(np.diag(displacement) - 1)) < eps:
            print('Converged after {} iterations'.format(it))
            break

    return X

def dist_to_weight(distance):
    return np.exp(-distance**2)

#def draw(mat):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2])

for idx, jdx in mol.edges():
    w = mol[idx][jdx].get('weight', 1)
    d = 1/w
    mol[idx][jdx]['distance'] = d
    mol[idx][jdx]['weight'] = dist_to_weight(d)

paths = []
for target in nx.all_pairs_shortest_path(mol, cutoff=2):
    for path in target[1].values():
        if len(path) == 3:
            paths.append(path)

mol_angle = mol.copy()
angle = np.deg2rad(180-109)
for idx, jdx, kdx in paths:
    l1 = mol[idx][jdx]['distance']
    l2 = mol[jdx][kdx]['distance']
    length = l1**2 * np.sin(angle)**2 + (l2 + l1*np.cos(angle))**2
    length = np.sqrt(length)
    mol_angle.add_edge(idx, kdx, distance=length, weight=dist_to_weight(length))
print('Added {} angles.'.format(len(paths)))

#embedded = MDS(mol_angle)
embedded = spectral(mol_angle)
embedded = smacof_dense(mol_angle, eps=2e-2, maxiter=5000, start=embedded)
#embedded = smacof(mol_angle, maxiter=5000)

orig_pos = np.array([mol.node[idx]['position'] for idx in mol.node])

for idx in mol.node:
    mol.node[idx]['position'] = embedded[idx]
draw(mol, node_size=30, with_label=True)
n = 5

edges = [random.choice(list(mol.edges())) for _ in range(n)]
print(edges)

for idx, jdx in edges:
#    d1 = np.linalg.norm(mol.node[idx]['position'] - mol.node[jdx]['position'])
    d1 = np.linalg.norm(orig_pos[idx] - orig_pos[jdx])
    d2 = np.linalg.norm(embedded[idx] - embedded[jdx])
    print(idx, jdx, d1, d2)
