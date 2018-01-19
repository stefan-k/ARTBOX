#!/usr/bin/env python3
"""Generate training data for the neural network

Author: Stefan Kroboth <stefan.kroboth@gmail.com>
"""

import os
import numpy as np
from file import ReconData
from operators import Operator
from cg import CG
from tools import save_image

os.environ['CUDA_DEVICE'] = str(2)


def main():
    result_dir = "/raid/groupspace/range/kroboth/neuralnetwork/"
    MODE = 2
    if MODE == 1:
        i_res_x = 128
        i_res_y = 128
        nT = i_res_x * i_res_y/8
        dims = 2
        MASK = 1
        SEM = 1
        RF = 1
        N_rand = 200
        N_traj = 100000
        CG_iter = 100
        directory = 'trainingdata/mode1/'
        result_dir = result_dir + "/mode1/"
        nC = 4
    if MODE == 2:
        i_res_x = 128
        i_res_y = 128
        nT = i_res_x * i_res_y/4
        dims = 2
        MASK = 1
        SEM = 1
        RF = 2
        N_rand = 200
        N_traj = 100000
        CG_iter = 100
        directory = 'trainingdata/mode2/'
        result_dir = result_dir + "/mode2/"
        nC = 8

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    sems = np.atleast_3d(np.zeros((0, i_res_x, i_res_y)))
    g = np.zeros((0, i_res_x, i_res_y, dims))
    x = np.linspace(-0.5, 0.5-1.0/i_res_x, i_res_x)
    y = np.linspace(-0.5, 0.5-1.0/i_res_y, i_res_y)
    X, Y = np.meshgrid(x, y)
    msk = np.ones((i_res_x, i_res_y))

    # MASK
    if MASK == 0:
        pass
    elif MASK == 1:
        msk[(np.power(X, 2) + np.power(Y, 2)) > 0.25] = 0
    else:
        raise ValueError('wrong MASK index')

    X = np.multiply(X, msk)
    Y = np.multiply(Y, msk)

    # SEMs
    if SEM == 1:
        sems = np.concatenate((sems, np.expand_dims(X, 0)), axis=0)
        sems = np.concatenate((sems, np.expand_dims(Y, 0)), axis=0)
        gt = np.concatenate((np.ones((1, i_res_x, i_res_y, 1)),
                             np.zeros((1, i_res_x, i_res_y, 1))),
                            axis=3)
        g = np.concatenate((g, gt), axis=0)
        gt = np.concatenate((np.zeros((1, i_res_x, i_res_y, 1)),
                             np.ones((1, i_res_x, i_res_y, 1))),
                            axis=3)
        g = np.concatenate((g, gt), axis=0)

    # RF
    rf = np.atleast_2d(np.zeros((0, i_res_x*i_res_y)))
    if RF == 1:
        for ii in (1, 3, 5, 7):
            fnamer = 'b1_real_res_{}_{}.cvs'.format(i_res_x, ii)
            fnamei = 'b1_imag_res_{}_{}.cvs'.format(i_res_x, ii)
            datar = np.genfromtxt('data/rf/' + fnamer, delimiter=',')
            datai = np.genfromtxt('data/rf/' + fnamei, delimiter=',')
            rf = np.concatenate((rf,
                                 np.expand_dims(datar.flatten() +
                                               1j * datai.flatten(), 0)),
                                axis=0)
        rf_sum = np.abs(rf.reshape((nC, i_res_x, i_res_y))).sum(0)/nC
    if RF == 2:
        for ii in (1, 2, 3, 4, 5, 6, 7, 8):
            fnamer = 'b1_trio_real_res_{}_{}.csv'.format(i_res_x, ii)
            fnamei = 'b1_trio_imag_res_{}_{}.csv'.format(i_res_x, ii)
            datar = np.genfromtxt('data/rf/' + fnamer, delimiter=',')
            datai = np.genfromtxt('data/rf/' + fnamei, delimiter=',')
            rf = np.concatenate((rf,
                                 np.expand_dims(datar.flatten() +
                                               1j * datai.flatten(), 0)),
                                axis=0)
        rf_sum = np.abs(rf.reshape((nC, i_res_x, i_res_y))).sum(0)/nC
    data = ReconData()
    data.set_SEM(sems)
    data.set_Cmat(rf)
    data.set_Gmat(g, np.asarray([1.0/i_res_x, 1.0/i_res_y, 1.0]))

    for i in range(2, N_traj):
        # create random traj
        if MODE == 1:
            k = np.random.random_integers(-100*i_res_x/2,100*i_res_x/2,(dims, nT))\
                    .astype(np.float)/float(100.)

            #  x = np.linspace(-i_res_x/2, i_res_x/2-1.0, i_res_x)
            #  x = x[0::2]
            #  y = np.linspace(-i_res_y/2, i_res_y/2-1.0, i_res_y)
            #  y = y[0::2]
            #  X, Y = np.meshgrid(x, y)
            #  k = np.concatenate((np.atleast_2d(X.flatten()),
            #                      np.atleast_2d(Y.flatten())), 0)

        if MODE == 2:
            k = np.random.random_integers(-100*i_res_x/2,100*i_res_x/2,(dims, nT))\
                    .astype(np.float)/float(100.)
            data.set_traj(2*np.pi*k)
        # save file
        data.save(directory + "traj_" + str(i), "/traj")

        res = np.zeros((i_res_x, i_res_y), dtype=np.complex64)
        for j in range(N_rand):
            print("Traj " + str(i) + " Run " + str(j))
            # create random noise
            recondata = 1*(np.random.randn(8*nT) + 1j * np.random.randn(8*nT))
            data.set_recondata(recondata)

            op = Operator(data, max_threads=128)

            # reconstruct noise
            out_dir = result_dir + "/traj_" + str(i) + "/" + str(j) + "/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cg = CG(op, data, out_dir, iters=CG_iter, save_images=False,
                    save_matlab=True, verbose=3)
            (im, tot_iters) = cg.run()

            res += np.divide(np.abs(im), rf_sum)
            save_image(np.abs(res), result_dir, "gfactor_" + str(i), "png")


if __name__ == '__main__':
    main()
