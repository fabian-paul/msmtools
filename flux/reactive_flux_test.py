'''
Created on Aug 14, 2014

@author: noe
'''

import unittest
import numpy as np

import api as msmapi
import pyemma.msm.analysis as msmana


class TestReactiveFluxFunctions(unittest.TestCase):
    def setUp(self):
        # 5-state toy system
        self.P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
                      [0.1,  0.75, 0.05, 0.05, 0.05],
                      [0.05,  0.1,  0.8,  0.0,  0.05],
                      [0.0,  0.2, 0.0,  0.8,  0.0],
                      [0.0,  0.02, 0.02, 0.0,  0.96]])
        self.A = [0]
        self.B = [4]
        self.I = [1,2,3]
        
        # REFERENCE SOLUTION FOR PATH DECOMP
        self.ref_committor = np.array([ 0.,          0.35714286,  0.42857143,  0.35714286,  1.        ])
        self.ref_backwardcommittor = np.array([ 1.         , 0.65384615,  0.53125    , 0.65384615,  0.        ])
        self.ref_grossflux = np.array([[ 0.,          0.00771792,  0.00308717,  0.        ,  0.        ],
                                 [ 0.,          0.        ,  0.00308717,  0.00257264,  0.00720339],
                                 [ 0.,          0.00257264,  0.        ,  0.        ,  0.00360169],
                                 [ 0.,          0.00257264,  0.        ,  0.        ,  0.        ],
                                 [ 0.,          0.        ,  0.        ,  0.        ,  0.        ]])
        self.ref_netflux = np.array([[  0.00000000e+00,   7.71791768e-03,   3.08716707e-03,   0.00000000e+00,    0.00000000e+00],
                                     [  0.00000000e+00,   0.00000000e+00,   5.14527845e-04,   0.00000000e+00,    7.20338983e-03],
                                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    3.60169492e-03],
                                     [  0.00000000e+00,   4.33680869e-19,   0.00000000e+00,   0.00000000e+00,    0.00000000e+00],
                                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    0.00000000e+00]])
        self.ref_totalflux =  0.0108050847458
        self.ref_kAB =  0.0272727272727
        self.ref_mfptAB = 36.6666666667
        
        self.ref_paths = [[0,1,4],[0,2,4],[0,1,2,4]]
        self.ref_pathfluxes = np.array([0.00720338983051, 0.00308716707022, 0.000514527845036])
        
        self.ref_paths_99percent = [[0,1,4],[0,2,4]]
        self.ref_pathfluxes_99percent = np.array([0.00720338983051, 0.00308716707022])
        self.ref_majorflux_99percent = np.array([[ 0.         , 0.00720339 , 0.00308717 , 0.         , 0.        ],
                                                 [ 0.         , 0.         , 0.         , 0.         , 0.00720339],
                                                 [ 0.         , 0.         , 0.         , 0.         , 0.00308717],
                                                 [ 0.         , 0.         , 0.         , 0.         , 0.        ],
                                                 [ 0.         , 0.         , 0.         , 0.         , 0.        ]])
        
        # Testing:
        self.tpt1 = msmapi.tpt(self.P, self.A, self.B)
        
        # 16-state toy system
        P2_nonrev = np.array([[0.5, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.2, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.3, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.1, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.5],
                              ])
        pstat2_nonrev = msmana.statdist(P2_nonrev)
        # make reversible
        C = np.dot(np.diag(pstat2_nonrev), P2_nonrev)
        Csym = C + C.T
        self.P2 = Csym / np.sum(Csym,axis=1)[:,np.newaxis]
        pstat2 = msmana.statdist(self.P2)
        self.A2 = [0,4]
        self.B2 = [11,15]
        self.coarsesets2 = [[2,3,6,7],[10,11,14,15],[0,1,4,5],[8,9,12,13],]
        
        # REFERENCE SOLUTION CG
        self.ref2_tpt_sets = [set([0, 4]), set([2, 3, 6, 7]), set([10, 14]), set([1, 5]), set([8, 9, 12, 13]), set([11, 15])]
        self.ref2_cgA = [0]
        self.ref2_cgI = [1,2,3,4]
        self.ref2_cgB = [5]
        self.ref2_cgpstat = np.array([ 0.15995388,  0.18360442,  0.12990937,  0.11002342,  0.31928127,  0.09722765])
        self.ref2_cgcommittor = np.array([ 0.         , 0.56060272 , 0.73052426 , 0.19770537 , 0.36514272 , 1.        ])
        self.ref2_cgbackwardcommittor = np.array([ 1.         , 0.43939728 , 0.26947574 , 0.80229463 , 0.63485728 , 0.        ])
        self.ref2_cggrossflux = np.array([[ 0.         , 0.         , 0.         , 0.00427986 , 0.00282259 , 0.        ],
                                          [ 0.         , 0          , 0.00234578 , 0.00104307 , 0.         , 0.00201899],
                                          [ 0.         , 0.00113892 , 0          , 0.         , 0.00142583 , 0.00508346],
                                          [ 0.         , 0.00426892 , 0.         , 0          , 0.00190226 , 0.        ],
                                          [ 0.         , 0.         , 0.00530243 , 0.00084825 , 0          , 0.        ],
                                          [ 0.         , 0.         , 0.         , 0.         , 0.         , 0.        ]])
        self.ref2_cgnetflux = np.array([[ 0.         , 0.         , 0.         , 0.00427986 , 0.00282259 , 0.        ],
                                        [ 0.         , 0.         , 0.00120686 , 0.         , 0.         , 0.00201899],
                                        [ 0.         , 0.         , 0.         , 0.         , 0.         , 0.00508346],
                                        [ 0.         , 0.00322585 , 0.         , 0.         , 0.00105401 , 0.        ],
                                        [ 0.         , 0.         , 0.0038766  , 0.         , 0.         , 0.        ],
                                        [ 0.         , 0.         , 0.         , 0.         , 0.         , 0.        ]])
        
        #Testing
        self.tpt2 = msmapi.tpt(self.P2, self.A2, self.B2)
    
    
    def test_nstates(self):
        self.assertEqual(self.tpt1.nstates, np.shape(self.P)[0])
    
    def test_A(self):
        self.assertEqual(self.tpt1.A, self.A)
    
    def test_I(self):
        self.assertEqual(self.tpt1.I, self.I)
    
    def test_B(self):
        self.assertEqual(self.tpt1.B, self.B)
    
    def test_flux(self):
        self.assertTrue(np.allclose(self.tpt1.flux, self.ref_netflux, rtol=1e-02, atol=1e-07))
    
    def test_netflux(self):
        self.assertTrue(np.allclose(self.tpt1.net_flux, self.ref_netflux, rtol=1e-02, atol=1e-07))
    
    def test_grossflux(self):
        self.assertTrue(np.allclose(self.tpt1.gross_flux, self.ref_grossflux, rtol=1e-02, atol=1e-07))
    
    def test_committor(self):
        self.assertTrue(np.allclose(self.tpt1.committor, self.ref_committor, rtol=1e-02, atol=1e-07))
    
    def test_forwardcommittor(self):
        self.assertTrue(np.allclose(self.tpt1.forward_committor, self.ref_committor, rtol=1e-02, atol=1e-07))
    
    def test_backwardcommittor(self):
        self.assertTrue(np.allclose(self.tpt1.backward_committor, self.ref_backwardcommittor, rtol=1e-02, atol=1e-07))
    
    def test_total_flux(self):
        self.assertTrue(np.allclose(self.tpt1.total_flux, self.ref_totalflux, rtol=1e-02, atol=1e-07))
    
    def test_rate(self):
        self.assertTrue(np.allclose(self.tpt1.rate, self.ref_kAB, rtol=1e-02, atol=1e-07))
        self.assertTrue(np.allclose(1.0/self.tpt1.rate, self.ref_mfptAB, rtol=1e-02, atol=1e-07))
    
    def test_pathways(self):
        # all paths
        (paths,pathfluxes) = self.tpt1.pathways()
        self.assertEqual(len(paths), len(self.ref_paths))
        for i in range(len(paths)):
            self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths[i])))
        self.assertTrue(np.allclose(pathfluxes, self.ref_pathfluxes, rtol=1e-02, atol=1e-07))
        # major paths
        (paths,pathfluxes) = self.tpt1.pathways(fraction = 0.99)
        self.assertEqual(len(paths), len(self.ref_paths_99percent))
        for i in range(len(paths)):
            self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths_99percent[i])))
        self.assertTrue(np.allclose(pathfluxes, self.ref_pathfluxes_99percent, rtol=1e-02, atol=1e-07))
    
    def test_major_flux(self):
        # all flux
        self.assertTrue(np.allclose(self.tpt1.major_flux(fraction=1.0), self.ref_netflux, rtol=1e-02, atol=1e-07))
        # 0.99 flux
        self.assertTrue(np.allclose(self.tpt1.major_flux(fraction=0.99), self.ref_majorflux_99percent, rtol=1e-02, atol=1e-07))
    
    def test_coarse_grain(self):
        (tpt_sets,cgRF) = self.tpt2.coarse_grain(self.coarsesets2)
        self.assertEqual(tpt_sets, self.ref2_tpt_sets)
        self.assertEqual(cgRF.A, self.ref2_cgA)
        self.assertEqual(cgRF.I, self.ref2_cgI)
        self.assertEqual(cgRF.B, self.ref2_cgB)
        self.assertTrue(np.allclose(cgRF.stationary_distribution, self.ref2_cgpstat))
        self.assertTrue(np.allclose(cgRF.committor, self.ref2_cgcommittor))
        self.assertTrue(np.allclose(cgRF.forward_committor, self.ref2_cgcommittor))
        self.assertTrue(np.allclose(cgRF.backward_committor, self.ref2_cgbackwardcommittor))
        self.assertTrue(np.allclose(cgRF.flux, self.ref2_cgnetflux))
        self.assertTrue(np.allclose(cgRF.net_flux, self.ref2_cgnetflux))
        self.assertTrue(np.allclose(cgRF.gross_flux, self.ref2_cggrossflux))


if __name__ == "__main__":
    unittest.main()