.. _installation_instruction:

========================
Installation instruction
========================

.. _dependencies:

Dependencies
============

OpenPIV would not have been possible if other great open source projects did not
exist. We make extensive use of code and tools that other people have created, so 
you should install them before you can use OpenPIV.

The dependencies are:

* `python <http://python.org/>`_
* `scipy <http://numpy.scipy.org/>`_
* `numpy <http://www.scipy.org/>`_
* `cython <http://cython.org/>`_

On all platforms, the binary Enthought Python Distribution (EPD) is recommended. 
Visit http://www.enthought.com

How to install the dependencies on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On a Linux platform installing these dependencies should be trick. Often, if not always, 
python is installed by default, while the other dependencies should appear in your package
manager.

How to install the dependencies on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On Windows all these dependencies, as well as several other useful packages, can be installed
using the Python(x,y) distribution, available at http://www.pythonxy.com/. Note: Install it in Custom Directories, 
without spaces in the directory names (i.e. Program Files are prohibited), e.g. C:\Pythonxy\


How to install the dependencies on a Mac
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The binary (32 or 64 bit) Enthought Python Distribution (EPD) is recommended.  Visit http://www.enthought.com 


Get OpenPIV source code!
========================
To get OpenPiv source code check our git repository store at `<https://github.com/alexlib/openpiv-python>`_. To download the source code on your machine, you may want to set up git on your computer, so that you can follow the development and get the latest version.  Please look at http://help.github.com/ which provide extensive help for how to set up git. If you are not confortable on setting up git you may want to download a `tarball <https://github.com/alexlib/openpiv-python/downloads>`_ containing the source code from the Github repository. 

To get the source code using git, clone our repository with the command::

    git clone http://github.com/alexlib/openpiv-python.git

and update from time to time with::

    git pull

Then, to install OpenPiv on your machine, run::
    
    python setupy.py install --prefix=$DIR
    
where ``$DIR`` is the folder you want to install OpenPIV in. If you want to install it system-wide omit the ``--prefix`` option, but you should have root priviles to do so. Remember to update the PYTHONPATH environment variable if you used a custom installation directory. If you downloaded the tarball you should run this command, too.

Having problems?
================
If you encountered some issues, found difficult to install OpenPIV following these instructions
please drop us an email to openpiv-develop@lists.sourceforge.net, so that we can help you and 
improve this page for other users. We also provide support on how to use OpenPiv, so you are welcome to send us a mail for any questions you may have.





