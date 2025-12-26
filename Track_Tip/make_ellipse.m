clear all
close all
clc

%% heterogeneities distribution (atria, Luther et al. 2011, Fig. 3a)

n = 1400;          % number of holes
alpha = 2.75;      % scaling exponent for atrial tissue
C = 0.015;         % normalization constant from your plot

% Physical cutoffs (from micro-CT data)
Rmin_mm = 0.06;    % 60 microns = 0.06 mm
Rmax_mm = 0.8;     % 0.8 mm

% Inverse transform sampling for power-law with finite upper cutoff
U = rand(n,1);
R_mm = ( Rmin_mm^(1-alpha) + U .* (Rmax_mm^(1-alpha) - Rmin_mm^(1-alpha)) ).^(1/(1-alpha));

% truncate at Rmax
R_mm(R_mm > Rmax_mm) = [];

% domain mapping: 10 cm = 100 mm mapped to 1000 pixels → 0.1 mm/pixel
px_size = 0.1;   % mm per pixel
R_px = round(R_mm / px_size);

fprintf('Generated %d holes\n', length(R_px));
fprintf('Min=%.3f mm, Max=%.3f mm, Mean=%.3f mm\n', ...
        min(R_mm), max(R_mm), mean(R_mm));

%% Plot the distribution (unchanged)
numBins = 12; 
edges = [0.06 0.07 0.08 0.09 0.10 0.2 0.3 0.4 0.5 0.6 0.7 0.8];

[counts, edges] = histcounts(R_mm, edges);
figure; 
histogram('BinEdges',edges,'BinCounts',counts);
title('Sampled radii histogram (mm)');

binCenters = sqrt(edges(1:end-1) .* edges(2:end));
counts(1:4) = counts(1:4) * 10;

figure;
loglog(binCenters, counts/100, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r')
hold on

alpha = 2.75;
xFit = logspace(log10(0.04), log10(10^(0.1)), 200);
yFit = C * xFit.^(-alpha);
loglog(xFit, yFit, 'k--', 'LineWidth', 2)

xlabel('Radius (mm)')
ylabel('Frequency')
title('Frequency vs. Radius (log-log)')
legend('Data (binned)', sprintf('C r^{-%0.2f}', alpha), 'Location', 'southwest')
grid on

ylim([1e-2 1e2])                        
xlim([0.04 10^(0.1)])                   
xticks([1e-3 1e-2 1e-1 1e0])
xticklabels({'10^{-3}','10^{-2}','10^{-1}','10^{0}'})
hold off

%% Setup grid
width = 512;
height = 512;
L = 100; % physical domain length in mm (10 cm)
x = linspace(0,L,width);
y = linspace(0,L,height);
[XX,YY] = meshgrid(x,y);

%% Helper function (inline) to stamp ONE ellipse into mask
% Inputs:
%   M          current mask
%   cx,cy      center in mm
%   a_mm,b_mm  ellipse semi-axes in mm (a = 10*b)
%   theta      rotation angle [rad]
%   XX,YY,x,y  grid info
% Output:
%   Mout       updated mask
ellipse_stamp = @(M,cx,cy,a_mm,b_mm,theta) ...
    local_stamp(M,cx,cy,a_mm,b_mm,theta,XX,YY,x,y);

function Mout = local_stamp(M,cx,cy,a_mm,b_mm,theta,XX,YY,x,y)
    % bounding box half-size (conservative) in mm
    halfbox = max(a_mm,b_mm);

    % restrict to region around ellipse
    x_min = max(cx - halfbox, x(1));
    x_max = min(cx + halfbox, x(end));
    y_min = max(cy - halfbox, y(1));
    y_max = min(cy + halfbox, y(end));

    % find indices in that window
    idx_x = find( x >= x_min & x <= x_max );
    idx_y = find( y >= y_min & y <= y_max );

    % grab subgrid
    Xsub = XX(idx_y, idx_x);
    Ysub = YY(idx_y, idx_x);

    % translate to ellipse center
    Xc = Xsub - cx;
    Yc = Ysub - cy;

    % rotate coords by -theta
    cosT = cos(theta);
    sinT = sin(theta);
    Xr =  cosT * Xc + sinT * Yc;
    Yr = -sinT * Xc + cosT * Yc;

    % inside test: (Xr/a)^2 + (Yr/b)^2 <= 1
    inside = (Xr./a_mm).^2 + (Yr./b_mm).^2 <= 1;

    % stamp
    Mlocal = M(idx_y, idx_x);
    Mlocal(inside) = 1;
    M(idx_y, idx_x) = Mlocal;

    Mout = M;
end

%% --- SQUARE DOMAIN ---
M = zeros(height,width);  % note: you used M=zeros(width,height) before; that transposes.
                          % I'll use [height,width] which matches meshgrid(y,x). 
                          % If you need old orientation for downstream code,
                          % swap back to M=zeros(width,height) and adjust indexing.

for rrr = 1:length(R_px)
    rad = R_px(rrr) * px_size;  % mm, "equivalent circle" radius

    %%% ELLIPSE CHANGE %%%
    % derive ellipse semi-axes in mm (a is major, b is minor)
    b_mm = rad / sqrt(10);
    a_mm = rad * sqrt(10);

    % pick random center that keeps ellipse fully in square domain
    % use 'a_mm' as conservative padding
    cx = a_mm + (L - 2*a_mm)*rand();
    cy = a_mm + (L - 2*a_mm)*rand();

    % random rotation
    theta = 2*pi*rand();

    % stamp ellipse
    M = ellipse_stamp(M,cx,cy,a_mm,b_mm,theta);
end    

% add boundary walls (set perimeter to 1)
M(:,1)=1; M(:,end)=1; M(1,:)=1; M(end,:)=1;

figure; imagesc(x,y,M); colormap('gray'); axis equal tight;
title('Square domain mask (ellipses)');
set(gca,'YDir','normal');

A = M;
Anew = double(A==0); % invert
figure; imagesc(x,y,Anew); colormap('gray'); axis equal tight;
title('Square inverted mask (ellipses)');
set(gca,'YDir','normal');

% (optional save for square domain if you want)
% AAA = cat(3, Anew, zeros(height,width,3)); % RGBA-ish
% AAA2 = permute(AAA, [3 1 2]);
% AAA3 = reshape(AAA2,1,height*width*4,1,1);
% masktexture = cat(2,[width height], AAA3);
% writematrix(masktexture, 'masksquare_ellipses.csv');

%% --- CIRCLE DOMAIN ---
M = zeros(height,width);

radius_domain = 0.5*0.99*L; % ~circle radius in mm
dist0 = sqrt((XX-0.5*L).^2+(YY-0.5*L).^2);
M(dist0 >= radius_domain) = 1; % outside big circle = wall

for rrr = 1:length(R_px)
    rad = R_px(rrr) * px_size; % mm

    %%% ELLIPSE CHANGE %%%
    b_mm = rad / sqrt(10);
    a_mm = rad * sqrt(10);

    % choose a center that keeps ellipse inside circular domain
    valid = false;
    while ~valid
        cx = rand()*L;
        cy = rand()*L;

        % distance of center from the big-circle center:
        d_center = sqrt((cx-0.5*L).^2 + (cy-0.5*L).^2);

        % require full ellipse to fit; again conservative using a_mm
        if (d_center + a_mm) <= radius_domain
            valid = true;
        end
    end

    theta = 2*pi*rand(); % random orientation

    % stamp ellipse into mask
    M = ellipse_stamp(M,cx,cy,a_mm,b_mm,theta);
end

figure; imagesc(x,y,M); colormap('gray'); axis equal tight;
title('Circle domain mask (ellipses)');
set(gca,'YDir','normal');

A = M;
Anew = double(A==0); % invert
figure; imagesc(x,y,Anew); colormap('gray'); axis equal tight;
title('Circle inverted mask (ellipses)');
set(gca,'YDir','normal');

%% --- SAVE CIRCLE MASK ---
AAA = cat(3, Anew, zeros(height,width,3));
AAA2 = permute(AAA, [3 1 2]);
AAA3 = reshape(AAA2,1,height*width*4,1,1);
masktexture = cat(2,[width height], AAA3);
writematrix(masktexture, 'maskcirclenew7_ellipses.csv');
