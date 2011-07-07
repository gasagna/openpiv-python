import unittest
import glob
import os

import numpy as np
import numpy.testing as npt

import openpiv.objects


# test data
files_a = sorted( glob.glob('data/2image_*0.tif'))
files_b = sorted( glob.glob('data/2image_*1.tif'))

class TestImagePair( unittest.TestCase ):
    def setUp(self):
        self.file_a = os.path.abspath( files_a[0] )
        self.file_b = os.path.abspath( files_b[0] )
        self.params = openpiv.objects.ProcessParameters()
        
    def test_init(self):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        self.assertTrue( ip.index == 1)
        self.assertTrue( ip.file_a == self.file_a )
        self.assertTrue( ip.file_b == self.file_b )
        self.assertTrue( ip.size == (1012, 1008) )
        self.assertTrue( isinstance(ip.frame_a, np.ndarray))
        
    def test_process_extended_search_area_piv_wo_s2n( self ):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        flow_field = ip.process( self.params )
        self.assertTrue( isinstance( flow_field, openpiv.objects.FlowField ) )
        
    def test_process_extended_search_area_piv_w_s2n( self ):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        # set computation of signal2noise ratio 
        self.params['sig2noise_method'] = 'peak2peak'
        flow_field = ip.process( self.params )
        self.assertTrue( isinstance( flow_field, openpiv.objects.FlowField ) )
        
    def test_process_extended_search_area_piv_w_raise( self ):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        # remove one key
        del self.params['method']
        self.assertRaises( openpiv.objects.PivParamsError, ip.process, self.params )
        
        
        
        
        
        
   

def suite():
    tests = []
    tests.append( unittest.TestLoader().loadTestsFromTestCase(TestImagePair) )
   
    return unittest.TestSuite( tests )
    
if __name__ == '__main__':
    unittest.main()
        
