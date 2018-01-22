.. ARTBOX documentation master file, created by
   sphinx-quickstart on Tue Jan 16 10:20:39 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. :caption: Contents:

ARTBOX documentation
====================

ARTBOX is a fast image reconstruction toolbox for magnetic resonance imaging (MRI).

Features:

* Conjugate gradients solver.
* Chambolle-Pock solver for Total Generalized Variation (TGV) regularized problems. (Based on code by `Kristian Bredies <https://imsc.uni-graz.at/bredies/index.html>`_.)
* Modified Chambolle-Pock solver for TGV regularized problems called TGV-CG.
* Uses the full encoding matrix as forward model. Therefore it is easy to include B0 inhomogeneities, non-Cartesian trajectories and nonlinear SEMs (spatially encoding magnetic fields, also called gradients).
* All operations heavily accelerated with GPU code using pycuda. 
* Modular design. Use the accelerated operators to implement your own algorithms (and ideally issue a pull request).

This piece of software is for *research use only* and *not for diagnostic use*!

Setup
-----

To install ARTBOX, simply clone the repository and `cd` into the directory.

.. code-block:: bash

   git clone git@github.com:stefan-k/ARTBOX.git
   cd ARTBOX

.. toctree::
   :maxdepth: 1

   reconfile
   operators
   tgv
   tgv_kernels
   cg
   tools


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
