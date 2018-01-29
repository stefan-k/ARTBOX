% Copyright 2018 Stefan Kroboth
%
% Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
% http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
% http://opensource.org/licenses/MIT>, at your option. This file may not be
% copied, modified, or distributed except according to those terms.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example code
%%
%% This code...
%%   1. ... creates a dataset for simulation
%%   2. ... runs a simulation using ARTBOX
%%   3. ... loads the simulated data
%%   4. ... adds some noise to the simulated data
%%   5. ... prepares dataset for reconstruction
%%   6. ... reconstructs image from data
%%   7. ... does a cleanup
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Resolution of reconstructed image (be aware that reconstruction time increases
% with this number)
res = 128;

% Simulation should be at a much higher resolution than reconstruction!
% 8*res will lead to 8*8 = 64 "spins per voxel"
sim_res = 8*res;

% ARTBOX path
artbox = '../../artbox-cli.py';

% RF sensitivity maps
rf_file = 'RF.mat';

% temporary directory
tmp_dir = '.tmp';

% GPU number
gpu_num = 0;

% No need to touch the following
if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create Dataset for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Trajectory
[t1, t2] = meshgrid(-res/2:1:(res/2-1));
traj_lin = [t1(:)'; t2(:)'];

% RF
rf = load(rf_file);
rf = interp_rf(rf.RF_MAPS_256, sim_res, sim_res);

% SEM
[X, Y] = meshgrid(linspace(-0.5, 0.5-1/sim_res, sim_res));
SEM = zeros(2, sim_res, sim_res);
SEM(1, :, :) = X;
SEM(2, :, :) = Y;

% Gmat
Gmat = zeros(2, sim_res, sim_res, 3);
Gmat(1, :, :, 1) = 1;
Gmat(2, :, :, 2) = 1;

% Create struct 'S'. Be aware that it has to be 'S'!
S = struct;
% S.k: [nF, nT]  # k-space trajectory
S.k = traj_lin;
% S.Cmat: [nC, nX1, nX2]  # RF coil sensitivity maps
S.Cmat = rf;
% S.SEM: [nF, nX1, nX2]  # spatial magnetic encoding fields
S.SEM = SEM;
% S.Gmat: [nF, nX1, nX2, 3]  # Derivative of SEMs
S.Gmat = Gmat;
% S.object: [nX1, nX2]  # object to be simulated
S.object = phantom(sim_res);


% Save the data. This must be in the v7 file format!
save('simustruct.mat', 'S', '-v7');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run simulation using ARTBOX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r = system(['python ' artbox ' --forward -o ' tmp_dir '/simu --gpu '...
            num2str(gpu_num) ' simustruct.mat -sm -y']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load simulated data, add noise and prepare for reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simu = load([tmp_dir '/simu/simustruct/mat/img_result.mat']);
sim_data = simu.img_result;

% TODO: Noise

% RF
rf = load(rf_file);
rf = interp_rf(rf.RF_MAPS_256, res, res);

% SEM
[X, Y] = meshgrid(linspace(-0.5, 0.5-1/res, res));
SEM = zeros(2, res, res);
SEM(1, :, :) = X;
SEM(2, :, :) = Y;


% Gmat
Gmat = zeros(2, res, res, 3);
Gmat(1, :, :, 1) = 1;
Gmat(2, :, :, 2) = 1;

% Create struct 'S'. Be aware that it has to be 'S'!
S = struct;
% S.k: [nF, nT]  # k-space trajectory
S.k = traj_lin;
% S.Cmat: [nC, nX1, nX2]  # RF coil sensitivity maps
S.Cmat = rf;
% S.SEM: [nF, nX1, nX2]  # spatial magnetic encoding fields
S.SEM = SEM;
% S.Gmat: [nF, nX1, nX2, 3]  # Derivative of SEMs
S.Gmat = Gmat;
% S.recondata: [nC, nT]  # measured/simulated data
S.recondata = sim_data.';


% Save the data. This must be in the v7 file format!
save('reconstruct.mat', 'S', '-v7');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run reconstruction using ARTBOX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r = system(['python ' artbox ' --cg -o ' tmp_dir '/recon --gpu '...
            num2str(gpu_num) ' reconstruct.mat -y -i 50']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load and show reconstructed images 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recon = load([tmp_dir '/recon/reconstruct/cg/mat/img_result.mat']);
recon_data = recon.img_result;

figure;
imshow(abs(recon_data), []);
