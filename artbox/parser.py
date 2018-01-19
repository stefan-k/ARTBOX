# Copyright 2018 Stefan Kroboth
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
"""Command line parser

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""
# pylint: disable=invalid-name

import argparse as ap

# INITIALIZE PARSER
parser = ap.ArgumentParser(description="ARTBOX: Accelerated Reconstruction " +
                           "Toolbox")

# PARSE DATA
parser.add_argument("data", help="Data file(s)", nargs='*')

# RECONSTRUCTION METHODS
recon = parser.add_argument_group("Reconstruction Methods",
                                  "Several methods in one call possible")
recon.add_argument("--tgv",
                   help="TGV regularized reconstruction with encoding matrix",
                   action="store_true")
recon.add_argument("--tgvcg",
                   help="TGV regularized reconstruction with inner CG",
                   action="store_true")
recon.add_argument("--cg",
                   help="CG based reconstruction using the encoding matrix",
                   action="store_true")

# PARAMETERS

# RECONSTRUCTION PARAMETERS
reconpar = parser.add_argument_group("Reconstruction Parameters",
                                     "Reconstruction parameters for all " +
                                     "methods.")
reconpar.add_argument("-i",
                      "--iters",
                      help="Number of iterations",
                      type=int,
                      default=500)
reconpar.add_argument("-ii",
                      "--inner-iters",
                      help="Number of inner CG iterations (default: 20)",
                      type=int,
                      default=20)
reconpar.add_argument("-a",
                      "--alpha",
                      help="TGV alpha parameter (default: 4e-5)",
                      type=float,
                      default=4e-5)
reconpar.add_argument("--fac",
                      help="TGV factor parameter (default: 2)",
                      type=float,
                      default=2)
reconpar.add_argument("--tau-p",
                      help="TGV tau_p parameter (default: 0.0625, will \
                      be ignored for inner CG methods)",
                      default=0.0625,
                      type=float)
reconpar.add_argument("--tau-d",
                      help="TGV tau_d parameter (default: 0.125, will \
                      be ignored for inner CG methods)",
                      default=0.125,
                      type=float)
reconpar.add_argument("--L2",
                      help="L2 parameter for TGV reconstruction \
                      (default: 11.828)",
                      default=None,
                      type=float)
reconpar.add_argument("--reduction",
                      help="TGV: Reduction factor for alpha \
                      (default: 2^-8)",
                      default=2**(-8),
                      type=float)
reconpar.add_argument("--norm-div",
                      help="norm_div parameter. Reduce estimated norm by this \
                      factor to increase convergence speed. Use carefully, \
                      might cause problems if too high. Will be ignored if \
                      --norm-est is passed. (default: 1)",
                      default=1,
                      type=float)
reconpar.add_argument("--norm-est",
                      help="Provide norm estimation (skips computation of \
                      norm_est)",
                      type=float,
                      default=None)
reconpar.add_argument("--norm-est-iters",
                      help="Number of iterations to perform for norm \
                      estimation (default: 10)",
                      default=10,
                      type=int)
reconpar.add_argument("--absolute-tolerance",
                      help="Absolute tolerance for CG (default: 1e-19)",
                      default=1e-19,
                      type=float)
reconpar.add_argument("--relative-tolerance",
                      help="Relative tolerance for CG (default: 1e-20)",
                      default=1e-20,
                      type=float)

# ENCODING MAT RECON
reconenc = parser.add_argument_group('Encoding Matrix Reconstruction \
                                     Parameters',
                                     'Encoding matrix specific reconstruction \
                                     parameters.')
reconenc.add_argument("--max-threads",
                      help="Maximum number of threads (default: 256)",
                      type=int,
                      default=256)
reconenc.add_argument("--hop",
                      help="Hop size for Encoding matrix forward and adjoint \
                      operator (default: 8192)",
                      type=int,
                      default=8192)
reconenc.add_argument("--hop-forward",
                      help="Hop size for Encoding matrix forward operator \
                      (overwrites hop parameter)",
                      type=int,
                      default=None)
reconenc.add_argument("--hop-adjoint",
                      help="Hop size for Encoding matrix adjoint operator \
                      (overwrites hop parameter)",
                      type=int,
                      default=None)
reconenc.add_argument("--hop-forward-sim",
                      help="Hop size for Encoding matrix forward operator for \
                      simulation (overwrites hop parameter)",
                      type=int,
                      default=None)
reconenc.add_argument("--divide",
                      help="Divide Encoding matrix forward and adjoint \
                      operators in # subproblems (default: 1)",
                      type=int,
                      default=1)
reconenc.add_argument("--divide-forward",
                      help="Divide Encoding matrix forward operator in # \
                      subproblems (overwrites divide parameter)",
                      type=int,
                      default=None)
reconenc.add_argument("--divide-adjoint",
                      help="Divide Encoding matrix adjoint operator in # \
                      subproblems (overwrites divide parameter)",
                      type=int,
                      default=None)
reconenc.add_argument("--divide-forward-sim",
                      help="Divide Encoding matrix forward operator for \
                      simulation in # subproblems (overwrites divide \
                      parameter)",
                      type=int,
                      default=8)
reconenc.add_argument("--show-kernel-params",
                      help="Show kernel call parameters",
                      action="store_true")

# SIMULATION WITH FORWARD MODEL
forward = parser.add_argument_group('Simulation (forward model)',
                                    'Forward model related parameters.')
forward.add_argument("--forward",
                     help="Apply forward model to object",
                     action="store_true")

# OUTPUT RELATED
output = parser.add_argument_group('Output',
                                   'Output related Parameters')
output.add_argument("-o",
                    "--out",
                    help="Output directory",
                    default="results")
output.add_argument("-sm",
                    "--save-matlab",
                    help="Save result as .mat file",
                    action="store_true")
output.add_argument("-si",
                    "--save-images",
                    help="Save all intermediate images",
                    action="store_true")
output.add_argument("-if", "--image-format",
                    help="Image extension (default: PNG)",
                    choices=['png', 'jpg', 'tiff', 'tif', 'gif'],
                    default='png')

timep = parser.add_argument_group('Time Measurement',
                                  'Time measurement of different parts of the \
                                  reconstruction process.')
timep.add_argument("--time",
                   help="Measure reconstruction time",
                   action="store_true")
timep.add_argument("--time-iters",
                   help="Measure reconstruction time for each iteration",
                   action="store_true")
timep.add_argument("--time-operator",
                   help="Measure time needed to build operators",
                   action="store_true")

# TEST ADJOINT
adj = parser.add_argument_group('Test Adjoint Operator',
                                'Test Encoding Matrix Operator.')
adj.add_argument("--test-adjoint-encoding-mat",
                 help="Test if the adjoint Operator is adjoint to the forward \
                 Operator (Encoding Matrix). You need to provide a dataset.",
                 action="store_true")


# GENERAL USABILLITY
usability = parser.add_argument_group('Usability',
                                      'Potentially useful tools.')
usability.add_argument("--no-progress",
                       help="Show iteration info instead of progress bar",
                       action="store_true",
                       default=False)
usability.add_argument("--gpu-info",
                       help="Print GPU information",
                       action="store_true")
usability.add_argument("--gpu",
                       help="Choose GPU device",
                       type=int,
                       default=0)
usability.add_argument("-y",
                       help="Do not ask questions, assume 'yes' for all \
                       (dangerous!)",
                       action="store_true")
usability.add_argument("--double",
                       help="Reconstruct/Simulate with double precision \
                       (EncodingMat only)",
                       action="store_true",
                       default=False)
noisyness = parser.add_mutually_exclusive_group()
noisyness.add_argument("-v",
                       "--verbose",
                       help="Show a lot of information",
                       action="count",
                       default=0)

# Parse arguments
args = parser.parse_args()
