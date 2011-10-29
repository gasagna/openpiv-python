.. OpenPiv documentation master file, created by
   sphinx-quickstart on Mon Apr 18 23:22:32 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenPiv: a python package for PIV image analysis.
=================================================

OpenPiv is a effort of scientists to deliver a toolbox of Particle Image velocimetry images using state-of-the-art algorithms. OpenPiv is written using the Python language, which means that is is flexible, powerful and completely platform independent. Openpiv is released under the `GPL Licence <http://en.wikipedia.org/wiki/GNU_General_Public_License>`_, which means that the source code is freely available for users to study, copy, modify and improve. This is important, since, because of its permissive licence, you are welcome to download, try and study OpenPiv source code for whatever needs you may have. Furthermore, you are encouraged to contribute to OpenPiv, with code, suggestions and critics.

Openpiv is mainly targeted at academics and student 

Features
========
 * fully open source code,
 * powerful and flexible scripting capabilities,
 * support for several of the most common image formats,
 * image batch processing, with parallel features for multicore machines,
 * zero order interrogation algorithms,
 * extensive outlier treatment,
 * data export in ASCII, HDF5 and Tecplot formats,
  * Comprehensive flow field post-processing: turbulence statistics, vorticity, deformation, velocity gradient tensor eigenvalues and invariants, two points temporal correlation, POD modes, clustering algorithms, vortex identification techcniques, ...
 * Full and comprehensive documentation in html or pdf format,

Quick Example
=============
This is a quick example to show off the powerful scripting capabilities
of Openpiv. This basic example shows how to process an image couple and save the result 
to a file, for post-processing it later.::

    # import openpiv namespace
   import openpiv
   
   # load default processing parameters
   process_parameters = openpiv.ProcessParameters()

   # load an image pair 
   image_pair = openpiv.ImagePair( 'file_a.tif', 'file_b.tif', index=0 )
   
   # process it
   flow_field = image_pair.process( process_parameters )
   
   # save data to file
   flow_field.save_to_file( '/home/user/data_directory' )
   
   
By adding a couple of lines more, flow field validation and other techniques
can be applied. You may want to check the :ref:_tutorial for other examples.

=========
Contents:
=========

.. toctree::
   :maxdepth: 1
   :titlesonly:

   src/installation_instruction
   src/downloads_page
   src/developers
   src/tutorial
   src/api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

