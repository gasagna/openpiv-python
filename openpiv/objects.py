"""The openpiv.objects module defines an object oriented interface for OpenPiv."""

__licence__ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__all__ = ['ImagePair', 'FlowField', 'Multiprocesser', 'PivParamsError', 'ProcessParameters']

import glob
import os.path
import multiprocessing
from UserDict import UserDict

import numpy as np
import scipy.misc

import openpiv.process
from openpiv.validation import global_std, global_val, local_median_val, sig2noise_val
import openpiv.lib

class ImagePair():
    """A class for holding data for two images."""
    def __init__ ( self, file_a, file_b, index ):
        """Create an image pair object."""
        self.file_a = file_a
        self.file_b = file_b
        
        self.frame_a = scipy.misc.imread( file_a, flatten=0).astype(np.int32)
        self.frame_b = scipy.misc.imread( file_b, flatten=0).astype(np.int32)

        self.index = index        
        self.size = self.frame_a.shape
        
    def process( self, process_parameters ):
        """Process an image pair."""
        # extract parameters specific of the processing algorithm
        try:
            if process_parameters.method == 'extended_search_area_piv':
                
                # call processing function
                out = openpiv.process.extended_search_area_piv( frame_a          = self.frame_a,
                                                                frame_b          = self.frame_b,
                                                                window_size      = process_parameters.window_size,
                                                                overlap          = process_parameters.overlap,
                                                                dt               = process_parameters.dt,
                                                                search_area_size = process_parameters.search_area_size,
                                                                subpixel_method  = process_parameters.subpixel_method,
                                                                sig2noise_method = process_parameters.sig2noise_method,
                                                                width            = process_parameters.width,
                                                                nfftx            = process_parameters.nfftx,
                                                                nffty            = process_parameters.nffty
                                                              )
                
                # unpack output depending on if we have s2n ratio too.
                if process_parameters.sig2noise_method:
                    u, v, s2n = out
                else:
                    u, v = out
                    s2n = None
                
        except KeyError, msg:
            raise PivParamsError("'%s' option is missing" % msg)
                             
        # get coordinates
        x, y = openpiv.process.get_coordinates(self.size, process_parameters.window_size, process_parameters.overlap )
        
        # build FlowField instance
        return FlowField(x, y, u, v, s2n)


class ProcessParameters():
    def __init__ ( self, config_file=None ):
        """A class working as a container for PIV processing parameters."""
        glb = {}
        loc = {}
        
        if config_file:
            execfile( config_file, glb, loc)
        else:
            execfile( openpiv.__default_process_parameters_file__, glb, loc)
            
        for k, v in loc.iteritems():
            setattr( self, k, v)
    
    def pretty_print(self):
        for key in sorted( dir( self ) ):
            if not key.startswith('__'):
                value = getattr( self, key )
                if not callable(value):
                    print '%s = %s' % ( str(key).rjust(30), str(value).ljust(30))


class FlowField():
    """A class for holding velocity data for validation/interpolation/postprocessing."""
    def __init__ ( self, x, y, u, v, s2n=None, index=0 ):
        """
        """
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.s2n = s2n
        self.mask = np.zeros(u.shape, dtype=bool)
        self.index = index
        
    def global_validation ( self, u_thresholds, v_thresholds ):
        """Validation step."""
        self.mask = global_val( self.u, self.v, u_thresholds, v_thresholds )
            
    def global_std_validation ( self, std_threshold ):
        """Validation step."""
        self.mask = global_std( self.u, self.v, std_threshold )
            
    def sig2noise_validation ( self, sn_threshold):
        """Validation step."""
        self.mask = sig2noise_val( self.u, self.v, self.s2n, sn_threshold )
            
    def local_median_validation ( self, u_threshold, v_threshold, size):
        """Validation step."""
        self.mask = local_median_val( self.u, self.v, u_threshold, v_threshold, size)
        
    def replace_outliers( self, max_iter=5, tol=1e-3, kernel_size=1 ):
        """Replace invalid vectors in the velocity field using an iterative image inpainting algorithm.
        
        The algorithm is the following:
        
        1) For each element in the arrays of the ``u`` and ``v`` components, replace it by a weighted average
           of the neighbouring elements which are not invalid themselves. The weights depends
           of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
           
        2) Several iterations are needed if there are adjacent invalid elements.
           If this is the case, inforation is "spread" from the edges of the missing
           regions iteratively, until the variation is below a certain threshold. 
        
        Parameters
        ----------
        max_iter : int
            the number of iterations of the inpainting algorithm
        tol : float
            the algorithm is stopped if root mean square variation between iteration
            is lower that this threshold. This number has physical significance of a velocity.
        kernel_size : int
            the size of the kernel, default is 1
        """
        # set outliers to nan, so that we can then replace them
        self.u[self.mask] = np.nan
        self.v[self.mask] = np.nan
        
        self.u = openpiv.lib.replace_nans( u, method='localmean', max_iter=max_iter, tol=tol, kernel_size=kernel_size )
        self.v = openpiv.lib.replace_nans( v, method='localmean', max_iter=max_iter, tol=tol, kernel_size=kernel_size )
        
    def save_to_file( self, datadir, file_pattern='field-%05d', fmt='%10.5f', process_parameters=None ):
        """Save flow field data to an ascii file."""
        
        # get name of the output file
        filename = os.path.join( os.path.abspath(datadir), file_pattern % self.index )
        
        # build output array
        out = np.vstack( [m.ravel() for m in [self.x, self.y, self.u, self.v, self.mask] ] )
                
        # open data file
        f = open(filename, 'w')

        if process_parameters:
            # save parameters to file
            params = '\n# '.join( ["%s = %s" % (k.rjust(30), repr(v).ljust(30)) for k, v in self.params ] )
            
            #write header
            header = """
            # Openpiv output data file: saved %s
            # Made with Openpiv version %s
            # 
            # Processing parameters
            # ---------------------
            # %s
            # 
            # Data
            # ----
            # x y u v mask""" % (time, version, params)
        
        np.savetxt( f, out.T, fmt=fmt )
        
    def rescale( self, scaling_factor ):
        """Apply an uniform scaling.
        
        Parameters
        ----------
        scaling_factor : float
            the image scaling factor in pixels per meter
        """
        for key in ['x', 'y', 'u', 'v']:
            value = getattr( self, key ) 
            value /= scaling_factor


class Multiprocesser():
    def __init__ ( self, data_dir, pattern_a, pattern_b):
        """A class to handle and process large sets of images.

        This class is responsible of loading image datasets
        and processing them. It has parallelization facilities
        to speed up the computation on multicore machines.
        
        It currently support only image pair obtained from 
        conventional double pulse piv acquisitions. Support 
        for continuos time resolved piv acquistion is in the 
        future.
        """
        
        # load lists of images 
        self.files_a = sorted( glob.glob( os.path.join( os.path.abspath(data_dir), pattern_a ) ) )
        self.files_b = sorted( glob.glob( os.path.join( os.path.abspath(data_dir), pattern_b ) ) )
        
        # number of images
        self.n_files = len(self.files_a)
        
        # check if everything was fine
        if not len(self.files_a) == len(self.files_b):
            raise ValueError('Something failed loading the image file. There should be an equal number of "a" and "b" files.')
            
        if not len(self.files_a):
            raise ValueError('Something failed loading the image file. No images were found. Please check directory and image pattern name.')

    def run( self, n_proc ):
        """Start to process images."""
        # create a list of tasks to be executed.
        image_pairs = [ (file_a, file_b, i) for file_a, file_b, i in zip( self.files_a, self.files_b, xrange(self.n_files) ) ]
        
        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        if n_proc > 1:
            pool = multiprocessing.Pool( processes = n_proc )
            res = pool.map( func, image_pairs )
        else:
            for image_pair in image_pairs:
                func( image_pair )


class PivParamsError(Exception):
    pass
