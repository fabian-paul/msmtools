
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""This module implements the connectivity functionality

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as csgraph
from six.moves import range


def connected_sets(C, directed=True):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        square matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in
        the corresponding connected component.

    """
    M = C.shape[0]
    """ Compute connected components of C. nc is the number of
    components, indices contain the component labels of the states
    """
    nc, indices = csgraph.connected_components(C, directed=directed, connection='strong')

    states = np.arange(M)  # Discrete states

    """Order indices"""
    ind = np.argsort(indices)
    indices = indices[ind]

    """Order states"""
    states = states[ind]
    """ The state index tuple is now of the following form (states,
    indices)=([s_23, s_17,...,s_3, s_2, ...], [0, 0, ..., 1, 1, ...])
    """

    """Find number of states per component"""
    count = np.bincount(indices)

    """Cumulative sum of count gives start and end indices of
    components"""
    csum = np.zeros(len(count) + 1, dtype=int)
    csum[1:] = np.cumsum(count)

    """Generate list containing components, sort each component by
    increasing state label"""
    cc = []
    for i in range(nc):
        cc.append(np.sort(states[csum[i]:csum[i + 1]]))

    """Sort by size of component - largest component first"""
    cc = sorted(cc, key=lambda x: -len(x))

    return cc


def largest_connected_set(C, directed=True):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        Count matrix specifying edge weights.

    Returns
    -------
    lcc : array of integers
        The largest connected component of the directed graph.

    """
    return connected_sets(C, directed=directed)[0]


def largest_connected_submatrix(C, directed=True, lcc=None):
    r"""Compute the count matrix of the largest connected set.

    The input count matrix is used as a weight matrix for the
    construction of a directed graph. The largest connected set of the
    constructed graph is computed. Vertices belonging to the largest
    connected component are used to generate a completely connected
    subgraph. The weight matrix of the subgraph is the desired
    completely connected count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        Count matrix specifying edge weights
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True
    lcc : (M,) ndarray, optional
       The largest connected set

    Returns
    -------
    C_cc : scipy.sparse matrix
        Count matrix of largest completely
        connected set of vertices (states)

    """
    if lcc is None:
        lcc = largest_connected_set(C, directed=directed)

    """Row slicing"""
    if scipy.sparse.issparse(C):
        C_cc = C.tocsr()
    else:
        C_cc = C
    C_cc = C_cc[lcc, :]

    """Column slicing"""
    if scipy.sparse.issparse(C):
        C_cc = C_cc.tocsc()
    C_cc = C_cc[:, lcc]

    if scipy.sparse.issparse(C):
        return C_cc.tocoo()
    else:
        return C_cc


def is_connected(C, directed=True):
    r"""Return true, if the input count matrix is completely connected.
    Effectively checking if the number of connected components equals one.

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    connected : boolean, returning true only if C is connected.


    """
    nc = csgraph.connected_components(C, directed=directed, connection='strong', \
                                      return_labels=False)
    return nc == 1