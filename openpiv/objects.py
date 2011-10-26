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
    def __init__ ( self, file_a, file_b, index ):
        """A class to hold data for two images.
        
        This is one of the base classes of openpiv, and one of the most 
        used. It is used for reading image files and processing them. 
        A wide ange of formats can be read.
        
        Parameters
        ----------
        file_a : str
            the path of first image file.
            
        file_b : str
            the path of second image file.
            
        index : int
            an integer number identifying the image couple.
            
        Attributes
        ----------
        file_a : str
            the path of first image file.
            
        file_b : str
            the path of second image file.
            
        index : int
            an integer number identifying the image couple.
            
        frame_a : numpy.ndarray, np.int32
            the array of grey levels of the first image
            
        frame_b : numpy.ndarray, np.int32
            the array of grey levels of the second image
            
        size : two elements tuple
            the shape of the images
        
        Methods
        -------
        process : process the image pair with the chosen processing parameters.
        
        Notes
        -----
        Images are loaded with the ``scipy.misc.imread`` function.
        
        Examples
        --------
        >>> from openpiv.objects import ImagePair
        >>> fa = 'data/image0312-a.tif'
        >>> fb = 'data/image0312-a.tif'
        >>> index = 312
        >>> im_pair = ImagePair( fa, fb, index )
        """
        self.file_a = file_a
        self.file_b = file_b
        
        self.frame_a = scipy.misc.imread( file_a, flatten=0 ).astype(np.int32)
        self.frame_b = scipy.misc.imread( file_b, flatten=0 ).astype(np.int32)

        self.index = index        
        self.size = self.frame_a.shape
        
    def process( self, process_parameters ):
        """Process an image pair. 
        
        The image pair is processed and the resulting vector field is returned.
        
        Parameters
        ----------
        process_parameters : an instance of :py:class:`openpiv.objects.ProcessParameters` class
            this is an object containing the parameters which will be used
            to process the image couple. See the documentation of this class for more details.
            
        Returns
        -------
        vec_field : an instance of the :py:class:`openpiv.objects.FlowField` class
            the vector field as computed by the processing algorithm.
        
        """
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
                                                                nfft             = process_parameters.nfft,
                                                              )
                
                # unpack output depending on if we have s2n ratio too.
                if process_parameters.sig2noise_method:
                    u, v, s2n = out
                else:
                    u, v = out
                    s2n = None
                
        except AttributeError, msg:
            raise PivParamsError("'%s' option is missing" % msg)
                             
        # get coordinates
        x, y = openpiv.process.get_coordinates(self.size, process_parameters.window_size, process_parameters.overlap )
        
        # build FlowField instance
        return FlowField(x, y, u, v, s2n)


class ProcessParameters():
    def __init__ ( self, config_file=None ):
        """A class working as a container for PIV processing parameters.
        
        Parameters
        ----------
        config_file : str, optional
            the path of the configuration file which will be loaded 
            and ised to process an image pair. If it is not given a 
            default set of parameters is used instead. 
        
        Notes
        -----
        Image pairs are processed by defining the algorithm and some other 
        parameters, such as, for example, the window size or the window overlap.
        This class is used to load parameters from custom configuration
        files and to hold them in a convenient form.
        
        Parameters are loaded form plain python files which must be
        ``SyntaxError``-free to be valid configuration files. 
        
        Configuration files contains a list of ``key = value`` pairs, 
        one for each line.  Several valid keys and value pairs exists; here
        only the most general are listed, while  algorithm-specific
        parameters can be found in the algorithms documentation page. FIXME
        
        * ``window_size`` : an integer number indicating the size 
          in pixel of the interrogation windows. Only square window 
          are available at the moment, so this is a single number. 
          The default value, if no ``config_file`` option is given, is 32.
          
        * `` overlap`` : the integer number of pixels by which two adjacent 
          interrogation window are overlapped. The overlap  must be less
          the ``window_size`` and greater or eqaul than zero.
        
        * ``dt`` : the time delay is seconds between the two frames
          of the image pair. This parametee is needed so that the output
          of the processing of the two images have units of pixels/seconds.
          
        * ``method`` : a string identifying the processing algorithm to be used.
          Valid options are, (at the moment), ``standard_piv`` and
          ``extended_search_area_piv``. For further details regarding the 
          algorithm details, please consult the Openpiv documentation FIXME.
          Each method has its own special parameters and therefore ignore 
          those parameters which is not made for. When the ``method``
          option is set, all the parameters related to the chosen 
          algorithm must be given, too, otherwise an error is raised.
        
        * ``subpixel_method`` : a string identifying the method
          used to estimate with subpixel accuracy the 
          correlation peak position. Valid values are 
          ``'centroid'``, ``'gaussian'`` and ``'parabolic'``.
          See FIXME pages for details on the subpixel peak location 
          algorithms.
          
        * ``sig2noise_method`` : a string identifying the method used to 
          compute a signal to noise ratio from the correlation map, a
          method which can be used to validate the vector field. Valid 
          values are ``'peak2peak'``, ``'peak2mean'`` and ``None``. 
          If ``None`` is given, no signal to noise computation is performed.
        
        * ``width`` : a integer number defining the half size of the 
          square region around the first correlation peak to ignore for
          finding the second peak, when computing the signal to noise 
          ratio from the correlation map. Default is 2; only used if 
          ``sig2noise_method=='peak2peak'``.
        
        * ``nfft`` : the size of the 2D FFT when computing the
          correlation map. Default is set to ``2 x window_size``.
        
        Examples
        --------
        An example configuration file can be::
        
            window_size = 32
            overlap = 0
            nfft = 64
            dt = 0.03
            method = 'extended_search_area_piv'
            search_area_size = 48
            subpixel_method  = 'centroid'
            sig2noise_method = None

            
        If this content is written inside a file, e.g., ``conf.cfg``,
        these paramters can be set into a ProcessParameters object
        with thise code::
        
        >>> import openpiv.objects
        >>> pp = openpiv.objects.ProcessParameters( 'conf.cfg' )
        
        Now the pp object has several new attributes. For example:
        >>> print pp.overlap
        ... 0
        >>> print pp.dt
        ... 0.03
        """
        glb = {}
        loc = {}
        
        try:
            if config_file:
                execfile( config_file, glb, loc)
            else:
                execfile( openpiv.__default_process_parameters_file__, glb, loc)
        except SyntaxError, msg:
            raise PivParamsError( msg )
                
        for k, v in loc.iteritems():
            setattr( self, k, v)
    
    def pretty_print( self ):
        """Prettily display the processing parameters.
        Mostly used in the interactive shell."""
        for key in sorted( dir( self ) ):
            if not key.startswith('__'):
                value = getattr( self, key )
                if not callable(value):
                    print '%s = %s' % ( str(key).rjust(30), str(value).ljust(30))


class FlowField():
    def __init__ ( self, x, y, u, v, s2n=None, index=0 ):
        """A class for holding velocity data for validation/interpolation/postprocessing.
        
        This class represents and hold information about the velocity 
        field as obtained from the processing of an image pair. Additionally, 
        this class can be instantiated from user defined velocity fields
        to post-process the data.
        
        Parameters
        ----------
        x : 2D numpy.ndarray
            a two dimensional array containing the ``x`` coordinates of the
            interrogation window centers, in pixels.
            
        y : 2D numpy.ndarray
            a two dimensional array containing the ``y`` coordinates of the
            interrogation window centers, in pixels.
    
        u : 2D numpy.ndarray
            a two dimensional array containing the ``u`` velocity component,
            in pixels/seconds.
            
        v : 2D numpy.ndarray
            a two dimensional array containing the ``v`` velocity component,
            in pixels/seconds.
            
        sig2noise : 2D numpy.ndarray or None, ( optional: only if sig2noise_method is not None )
            a two dimensional array the signal to noise ratio for each
            window pair.
            
        index : int, optional
            an integer number identifying the image couple from which 
            the flow field was obtained.
            
        Attributes
        ----------
        x : 2D numpy.ndarray
            a two dimensional array containing the ``x`` coordinates of the
            interrogation window centers, in pixels.
            
        y : 2D numpy.ndarray
            a two dimensional array containing the ``y`` coordinates of the
            interrogation window centers, in pixels.
    
        u : 2D numpy.ndarray
            a two dimensional array containing the ``u`` velocity component,
            in pixels/seconds.
            
        v : 2D numpy.ndarray
            a two dimensional array containing the ``v`` velocity component,
            in pixels/seconds.
            
        sig2noise : 2D numpy.ndarray or None, ( optional: only if sig2noise_method is not None )
            a two dimensional array the signal to noise ratio for each
            window pair.
            
        index : int, optional
            an integer number identifying the image couple from which 
            the flow field was obtained.
        
        mask : 2D numpy.ndarray of boolean values
            this array is used to mark vector as spurious. If the element
            ``i,j`` of the mask array is ``True``, then the corresponding 
            vector is regarded as spurious. Initially all elements are 
            set to ``False``, (no spurious vector). By applying validation
            methods some elements are then set to True.
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
