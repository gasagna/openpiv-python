import unittest
import glob
import os

import numpy as np
import numpy.testing as npt

import openpiv.objects


# test data
files_a = sorted( glob.glob('data/2image_*0.tif'))
files_b = sorted( glob.glob('data/2image_*1.tif'))

test_config_file = 'data/test-config.cfg'
bugged_test_config_file = 'data/bugged-test-config.cfg'

class TestProcessParameters( unittest.TestCase ):
    def setUp(self):
        self.params = openpiv.objects.ProcessParameters( config_file=test_config_file )
        
    def test_attributes( self ):
        """ Test whether attribtue are set"""
        self.assertTrue( self.params.window_size == 32 )
        self.assertTrue( self.params.overlap == 16 )
        self.assertTrue( self.params.dt == 0.001 )
        self.assertTrue( self.params.method == 'extended_search_area_piv' )
        self.assertTrue( self.params.search_area_size == 64 )
        self.assertTrue( self.params.subpixel_method  == 'gaussian' )
        self.assertTrue( self.params.sig2noise_method == None )
        self.assertTrue( self.params.width == 2 )
        self.assertTrue( self.params.nfftx == 64 )
        self.assertTrue( self.params.nffty == 64 )
    
    def test_assert_raise_piv_params_error( self ):
        self.assertRaises( openpiv.objects.PivParamsError, openpiv.objects.ProcessParameters, bugged_test_config_file  )

class TestImagePair( unittest.TestCase ):
    def setUp(self):
        self.file_a = os.path.abspath( files_a[0] )
        self.file_b = os.path.abspath( files_b[0] )
        self.params = openpiv.objects.ProcessParameters( config_file=test_config_file )
        
    def test_init(self):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        self.assertTrue( ip.index == 1)
        self.assertTrue( ip.file_a == self.file_a )
        self.assertTrue( ip.file_b == self.file_b )
        self.assertTrue( ip.size == (1012, 1008) )
        self.assertTrue( isinstance(ip.frame_a, np.ndarray))
        
    def test_process_extended_search_area_piv_wo_s2n( self ):
        """Test that class method computes the flow field"""
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        flow_field = ip.process( self.params )
        self.assertTrue( isinstance( flow_field, openpiv.objects.FlowField ) )
        
    def test_process_extended_search_area_piv_w_s2n( self ):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        # set computation of signal2noise ratio 
        self.params.sig2noise_method = 'peak2peak'
        flow_field = ip.process( self.params )
        self.assertTrue( isinstance( flow_field, openpiv.objects.FlowField ) )
        
    def test_process_extended_search_area_piv_w_raise( self ):
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        # remove one key
        del self.params.method
        self.assertRaises( openpiv.objects.PivParamsError, ip.process, self.params )
class TestFlowField( unittest.TestCase ):
    def setUp(self):
        self.file_a = os.path.abspath( files_a[0] )
        self.file_b = os.path.abspath( files_b[0] )
        self.params = openpiv.objects.ProcessParameters()
    
    def test_init_w_s2n( self ):
        """Test that attribute of signal to noise ratio is present."""
        ip = openpiv.objects.ImagePair( self.file_a, self.file_b, index=1 )
        # set computation of signal2noise ratio 
        self.params.sig2noise_method = 'peak2peak'
        flow_field = ip.process( self.params )
        self.assertTrue(
        
        
        
        
   

def suite():
    tests = []
    tests.append( unittest.TestLoader().loadTestsFromTestCase(TestImagePair) )
    tests.append( unittest.TestLoader().loadTestsFromTestCase(TestProcessParameters) )
   
    return unittest.TestSuite( tests )
    
if __name__ == '__main__':
    unittest.main()
        
