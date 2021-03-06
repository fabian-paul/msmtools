
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

'''
Created on Jul 23, 2014

@author: noe
'''
from __future__ import absolute_import
from __future__ import division

import numpy as np
import scipy
from scipy.stats import rv_discrete
import random
import six
from six.moves import range


# By FN
def number_of_states(dtrajs):
    r"""
    Determine the number of states from a set of discrete trajectories

    Parameters
    ----------
    dtrajs : list of int-arrays
        discrete trajectories
    """
    # determine number of states n
    nmax = 0
    for dtraj in dtrajs:
        nmax = max(nmax, np.max(dtraj))
    # return number of states
    return nmax + 1


def determine_lengths(dtrajs):
    r"""
    Determines the lengths of all trajectories

    Parameters
    ----------
    dtrajs : list of int-arrays
        discrete trajectories
    """
    if (isinstance(dtrajs[0], (int))):
        return len(dtrajs) * np.ones((1))
    lengths = np.zeros((len(dtrajs)))
    for i in range(len(dtrajs)):
        lengths[i] = len(dtrajs[i])
    return lengths


def bootstrap_trajectories(trajs, correlation_length):
    """
    Generates a randomly resampled count matrix given the input coordinates.

    See API function for full documentation.
    """
    # if we have just one trajectory, put it into a one-element list:
    if (isinstance(trajs[0], (int, int, float))):
        trajs = [trajs]
    ntraj = len(trajs)

    # determine correlation length to be used
    lengths = determine_lengths(trajs)
    Ltot = np.sum(lengths)
    Lmax = np.max(lengths)
    if (correlation_length < 1):
        correlation_length = Lmax

        # assign probabilites to select trajectories
    w_trajs = np.zeros((len(trajs)))
    for i in range(ntraj):
        w_trajs[i] = len(trajs[i])
    w_trajs /= np.sum(w_trajs)  # normalize to sum 1.0
    distrib_trajs = rv_discrete(values=(list(range(ntraj)), w_trajs))

    # generate subtrajectories
    Laccum = 0
    subs = []
    while (Laccum < Ltot):
        # pick a random trajectory
        itraj = distrib_trajs.rvs()
        # pick a starting frame
        t0 = random.randint(0, max(1, len(trajs[itraj]) - correlation_length))
        t1 = min(len(trajs[itraj]), t0 + correlation_length)
        # add new subtrajectory
        subs.append(trajs[itraj][t0:t1])
        # increment available states
        Laccum += (t1 - t0)

    # and return
    return subs


def bootstrap_counts_singletraj(dtraj, lagtime, n):
    """
    Samples n counts at the given lagtime from the given trajectory
    """
    # check if length is sufficient
    L = len(dtraj)
    if (lagtime > L):
        raise ValueError(
            'Cannot sample counts with lagtime ' + str(lagtime) + ' from a trajectory with length ' + str(L))
    # sample
    I = np.random.random_integers(0, L - lagtime - 1, size=n)
    J = I + lagtime

    # return state pairs
    return (dtraj[I], dtraj[J])


def bootstrap_counts(dtrajs, lagtime, corrlength=None):
    """
    Generates a randomly resampled count matrix given the input coordinates.

    See API function for full documentation.
    """
    # if we have just one trajectory, put it into a one-element list:
    if (isinstance(dtrajs[0], six.integer_types)):
        dtrajs = [dtrajs]
    ntraj = len(dtrajs)

    # can we do the estimate?
    lengths = determine_lengths(dtrajs)
    Lmax = np.max(lengths)
    Ltot = np.sum(lengths)
    if (lagtime >= Lmax):
        raise ValueError('Cannot estimate count matrix: lag time '
                         + str(lagtime) + ' is longer than the longest trajectory length ' + str(Lmax))

    # how many counts can we sample?
    if corrlength is None:
        corrlength = lagtime
    nsample = int(Ltot / corrlength)

    # determine number of states n
    n = number_of_states(dtrajs)

    # assigning trajectory sampling weights
    w_trajs = np.maximum(0.0, lengths - lagtime)
    w_trajs /= np.sum(w_trajs)  # normalize to sum 1.0
    distrib_trajs = rv_discrete(values=(list(range(ntraj)), w_trajs))
    # sample number of counts from each trajectory
    n_from_traj = np.bincount(distrib_trajs.rvs(size=nsample), minlength=ntraj)

    # for each trajectory, sample counts and stack them
    rows = np.zeros((nsample))
    cols = np.zeros((nsample))
    ones = np.ones((nsample))
    ncur = 0
    for i in range(len(n_from_traj)):
        if n_from_traj[i] > 0:
            (r, c) = bootstrap_counts_singletraj(dtrajs[i], lagtime, n_from_traj[i])
            rows[ncur:ncur + n_from_traj[i]] = r
            cols[ncur:ncur + n_from_traj[i]] = c
            ncur += n_from_traj[i]
    # sum over counts
    Csparse = scipy.sparse.coo_matrix((ones, (rows, cols)), shape=(n, n))

    return Csparse.tocsr()