% Copyright 2018 Stefan Kroboth
%
% Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
% http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
% http://opensource.org/licenses/MIT>, at your option. This file may not be
% copied, modified, or distributed except according to those terms.

function rf_out = interp_rf(rf, x, y)
    % rf_out = interp(rf, x, y)
    % 
    % Interpolates RF sensitivity maps of shape [nX1, nX2, nC] to [nC, x, y].
    % For convience (or confusion), the dimensions are permuted.
    % Interpolation is done separately for real and imaginary parts of the
    % array. 
    %
    % Author: Stefan Kroboth <stefan.kroboth@gmail.com>
    nC = size(rf, 3);
    [X, Y] = meshgrid(linspace(-1, 1, size(rf, 1)), ...
                      linspace(-1, 1, size(rf, 2)));
    [Xo, Yo] = meshgrid(linspace(-1, 1, x), ...
                        linspace(-1, 1, y));

    rf_out = zeros(nC, x, y);
    for ii = 1:nC
        rf_out(ii, :, :) = griddata(X, Y, ...
	                            real(rf(:, :, ii)), ...
				    Xo, Yo) + ...
                           griddata(X, Y, ...
	                            imag(rf(:, :, ii)), ...
				    Xo, Yo)*1i;
    end 
end
% function b1_mat=load_b1_maps(filename,x,y)
% 
%     rf = load(filename);
%     b1_mat_full = rf.RF_MAPS_256;
%     nC = size(b1_mat_full,3);
%     [X,Y]=meshgrid(linspace(-1,1,256));
%     
%     b1_mat_full = reshape(b1_mat_full,[],nC);
%     nPixel=numel(x);
%     b1_mat = zeros(nPixel,nC);
%     for i=1:nC
%         b1_mat(:,i) = griddata(X(:),Y(:),b1_mat_full(:,i),x,y);
%     end
%     
% end
