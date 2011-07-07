__doc__="""The openpiv.objects module defines an object oriented interface for OpenPiv."""
__licence_ = """
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

import glob
import sys
import os.path
import multiprocessing
from UserDict import UserDict
import ConfigParser

import numpy as np
import scipy.misc
import matplotlib.pyplot as pl

import openpiv.process
import openpiv.validation
import openpiv.lib

class ImagePair():
    """A class for holding data for two images."""
    def __init__ ( self, file_a, file_b, index ):
        """Create an image pair object."""
        self.file_a = file_a
        self.file_b = file_b
        
        self.frame_a = imread( file_a )
        self.frame_b = imread( file_b )

        self.index = index        
        self.size = self.frame_a.shape
        
    def process( self, parameters ):
        """Process an image pair."""
        # extract parameters specific of the processing algorithm
        try:
            if parameters['method'] == 'extended_search_area_piv':
                
                # call processing function
                out = openpiv.process.extended_search_area_piv( frame_a          = self.frame_a,
                                                                frame_b          = self.frame_b,
                                                                window_size      = parameters['window_size'],
                                                                overlap          = parameters['overlap'],
                                                                dt               = parameters['dt'],
                                                                search_area_size = parameters['search_area_size'],
                                                                subpixel_method  = parameters['subpixel_method'],
                                                                sig2noise_method = parameters['sig2noise_method'],
                                                                width            = parameters['width'],
                                                                nfftx            = parameters['nfftx'],
                                                                nffty            = parameters['nffty']
                                                              )
                
                # unpack output depending on if we have s2n ratio too.
                if parameters['sig2noise_method']:
                    u, v, s2n = out
                else:
                    u, v = out
                    s2n = None
                
        except KeyError, msg:
            raise PivParamsError("'%s' option is missing" % msg)
                             
        # get coordinates
        x, y = openpiv.process.get_coordinates(self.size, parameters['window_size'], parameters['overlap'] )
        
        return FlowField(x, y, u, v, parameters, s2n)

class FlowField():
    """A class for holding velocity data for validation/interpolation/postprocessing."""
    def __init__ ( self, x, y, u, v, parameters, s2n=None, ):
        """
        """
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.s2n = s2n
        self.mask = np.zeros(u.shape, dtype=bool)
        self.parameters
        
    def validate ( self, method, u_thresholds=None, v_thresholds=None, std_threshold=None, sn_threshold=None, u_threshold=None, v_threshold=None, size=None):
        """Validation step."""
        
        if method=='global':
            if not u_thresholds and v_thresholds:
                raise ValueError("global validation requires options 'u_thresholds' and 'v_thresholds' to be given")
            self.mask = openpiv.validation.global_val( self.u, self.v, u_thresholds, v_thresholds )
            
        if method=='global_std':
            if not std_threshold:
                raise ValueError("global std validation requires option 'std_threshold' to be given")
            self.mask = openpiv.validation.global_std( self.u, self.v, std_theshold )
            
        if method=='sig2noise':
            if not self.s2n:
                raise ValueError("signoise validation requires signal-to-noise ration information, which was not computed initially")
            if not sn_threshold:
                raise ValueError("signoise validation requires option 'sn_threshold' to be given")
            self.mask = openpiv.validation.sig2noise_val( self.u, self.v, self.s2n, sn_threshold )
            
        if method=='local_median':
            if not u_threshold and v_threshold and size:
                raise ValueError("local median validation requires options 'u_threshold', 'v_threshold' and 'size' to be given")
            self.mask = openpiv.validation.local_median_val( self.u, self.v, u_threshold, v_threshold, size)
        
    def replace_outliers( max_iter=5, tol=1e-3, kernel_size=1 ):
        """Replace invalid vectors in the  velocity field using an iterative image inpainting algorithm.
        
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
        
    def save_to_file( filename, fmt='%8.4f', delimiter='\t' ):
        """Save flow field data to an ascii file.
        
        Parameters
        ----------
        filename : string
            the path of the file where to save the flow field
            
        fmt : string
            a format string. See documentation of numpy.savetxt
            for more details.
        
        delimiter : string
            character separating columns
        """
        # build output array
        out = np.vstack( [m.ravel() for m in [x, y, u, v, mask] ] )
                
        # open data file
        f = open(filename, 'r')

        # save parameters to file
        params = '\n# '.join( ["%s = %s" % (k.rjust(30), repr(v).ljust(30)) for k, v in self. ] 
        
        #write header
        header = """
        # Openpiv output data file: saved %s
        # Made with Openpiv version %s
        # 
        # Processing parameters
        # ---------------------
        # 
        # x y u v mask""" % (time, version, params)
        
        np.savetxt( filename, out.T, fmt=fmt, delimiter=delimiter )

    
class ProcessParameters( UserDict ):
    def __init__ (self, config_file='' ):
        """
        This class provide a consistent way to set and get all the
        parameters needed to process piv image files. It uses a 
        ConfigParser.SafeConfigParser, which bring some cool features, 
        like reading parameters from INI files.  COnfigParser is a 
        built-in python module. See its documentation for more details.
        
        To instantiate this class you may want to provide as an argument 
        a specific configuration file, but default values are already loaded
        as you create an instance.
        
        The optional file should be formatted like an INI file, with a unique 
        section named 'options'. Example:
        
        [options]
        initial_window_size = 32
        final_window_size = 32
        ...
    
        Parameters
        ----------
        
        config_file : string, optional
            an optional configuration file containing user setting 
            processing parameters.
            
        """
        
        # instantiate ancestor's class
        configparser = ConfigParser.SafeConfigParser()
        
        # read configuration files
        configparser.read( [openpiv.__default_config_file__, config_file] )
        
        UserDict.__init__(self, configparser.items('options') )
        
        # cast parameters to the right type.
        # this trick is necessary because the ConfigParser
        # class only understands strings
        for k, v in self.iteritems():
            try:
                self[k] = int(v)
            except ValueError:
                try:
                    self[k] = float(v)
                except ValueError:
                    pass
            if v == 'None':
                self[k] = None
    
    def pretty_print ( self ):
        """
        Pretty print all the processing parameters.
        """
        for k, v in self.iteritems():
            print "%s = %s" % ( k.rjust(30), repr(v).ljust(30) )

class Hdf5Database( ):
    """
    A class for writing/reading PIV data to an hdf5 file.
    
    This is currently work in progress, because it is not easy to 
    make parallel write to an hdf file with h5py.
    """
    def __init__ ( self, database, mode='create' ):
        """
        A database can be opened in read/create/modes.
        """

        if mode in [ 'create', 'c', 'w']:
            try:
                self._fh = h5py.File( database, mode='w-' )
            except:
                msg = "A file names %s already exists in folder %s. Delete it first if you really want to." % ( os.path.basename(database), os.path.dirname(database) )
                raise IOError(msg)
        elif mode in [ 'read', 'r']:
            self._fh = h5py.File( os.path.abspath(database), mode='r' )
        else:
            raise ValueError('wrong mode. create of read are accepted')
            
        # two groups for the two velocity components
        self._fh.create_group('/u')
        self._fh.create_group('/v')

    def write_coordinates( self, x, y ):
        """
        Write two datasets with the coordinates of the PIV vectors
        """
        self._fh.create_dataset ( 'x', data = x )
        self._fh.create_dataset ( 'y', data = y )

    def write_velocity_field ( self, i, u, v ):
        """
        Write datasets for the two velocity components.
        """
        self._fh.create_dataset ( '/u/field%05d' % i, data = u )
        self._fh.create_dataset ( '/v/field%05d' % i, data = v )

    def close ( self ):
        """Close file"""
        self._fh.close()

class Multiprocesser():
    def __init__ ( self, data_dir, pattern_a, pattern_b  ):
        """A class to handle and process large sets of images.

        This class is responsible of loading image datasets
        and processing them. It has parallelization facilities
        to speed up the computation on multicore machines.
        
        It currently support only image pair obtained from 
        conventional double pulse piv acquisition. Support 
        for continuos time resolved piv acquistion is in the 
        future.
        
        
        Parameters
        ----------
        data_dir : str
            the path where image files are located 
            
        pattern_a : str
            a shell glob patter to match the first 
            frames.
            
        pattern_b : str
            a shell glob patter to match the second
            frames.

        Examples
        --------
        >>> multi = openpiv.tools.Multiprocesser( '/home/user/images', 'image_*_a.bmp', 'image_*_b.bmp')
    
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
            raise ValueError('Something failed loading the image file. No images were found. Please check directory and image template name.')

    def run( self, func, n_cpus=1 ):
        """Start to process images.
        
        Parameters
        ----------
        
        func : python function which will be executed for each 
            image pair. See tutorial for more details.
        
        n_cpus : int
            the number of processes to launch in parallel.
            For debugging purposes use n_cpus=1
        
        """

        # create a list of tasks to be executed.
        image_pairs = [ (file_a, file_b, i) for file_a, file_b, i in zip( self.files_a, self.files_b, xrange(self.n_files) ) ]
        
        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        if n_cpus > 1:
            pool = multiprocessing.Pool( processes = n_cpus )
            res = pool.map( func, image_pairs )
        else:
            for image_pair in image_pairs:
                func( image_pair )

class PivParamsError(Exception):
    pass
 

def display_vector_field( filename,**kw):
    """ Displays quiver plot of the data stored in the file 
    
    
    Parameters
    ----------
    filename :  string
        the absolute path of the text file
    
    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]
    
    
    See also:
    ---------
    matplotlib.pyplot.quiver
    
        
    Examples
    --------
    
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, width=0.0025) 

    
    """
    
    a = np.loadtxt(filename)
    pl.figure()
    pl.hold(True)
    invalid = a[:,4].astype('bool')
    
    valid = ~invalid
    pl.quiver(a[invalid,0],a[invalid,1],a[invalid,2],a[invalid,3],color='r',**kw)
    pl.quiver(a[valid,0],a[valid,1],a[valid,2],a[valid,3],color='b',**kw)
    pl.draw()
    pl.show()

def imread( filename ):
    """Read an image file into a numpy array
    using scipy.misc.imread
    
    Parameters
    ----------
    filename :  string
        the absolute path of the image file 
        
    Returns
    -------
    frame : np.ndarray
        a numpy array with grey levels
        
        
    Examples
    --------
    
    >>> image = openpiv.tools.imread( 'image.bmp' )
    >>> print image.shape 
        (1280, 1024)
    
    
    """
    
    return scipy.misc.imread( filename, flatten=0).astype(np.int32)


def display( message ):
    """Display a message to standard output.
    
    Parameters
    ----------
    message : string
        a message to be printed
    
    """
    sys.stdout.write(message)
    sys.stdout.write('\n')
    sys.stdout.flush()
