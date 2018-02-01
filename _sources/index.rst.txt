.. ARTBOX documentation master file, created by
   sphinx-quickstart on Tue Jan 16 10:20:39 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. :caption: Contents:

ARTBOX documentation
====================

ARTBOX is a fast image reconstruction toolbox for magnetic resonance imaging (MRI).

The operator implemented in this work allows you to incorporate B0 inhomogeneities, non-Cartesian trajectories and even non-linear encoding fields into MR image reconstruction/simulation. 
It furthermore considers the reconstructed pixels to be box functions instead of Diracs. This makes it possible to model intra-voxel dephasing which may improve the conditioning of the inverse.

Several image reconstruction methods (CG, TGV-regularization) are implemented.

Note: Please be aware that parts of this software may be refactored and therefore the API may change.

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

To install ARTBOX, simply clone the repository, `cd` into the directory and install the requirements via pip.

.. code-block:: bash

   git clone git@github.com:stefan-k/ARTBOX.git
   cd ARTBOX
   pip3 install < requirements.txt

It is recommended to use a virtual environment. The following code block shows how to use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest>`_ to set up a virtual environment:

.. code-block:: bash

   git clone git@github.com:stefan-k/ARTBOX.git
   cd ARTBOX
   mkvirtualenv --python=$(which python3) artbox
   pip3 install < requirements.txt

Make sure virtualenvwrapper is configured correctly.
Once the virtual environment is created, it needs to be activated in every shell where ARTBOX code should run:

.. code-block:: bash

   workon artbox


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
