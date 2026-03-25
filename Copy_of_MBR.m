
%  One falling ball + SINDy
%  Re-dependent drag
% =============================================
clear; clc; close all;

%% --- Physical / fluid constants ---
g   = 9.81;          % m/s^2, downward
rho = 1.211;         % kg/m^3, air density (as in paper)
mu  = 1.81e-5;       % Pa*s, dynamic viscosity

H0  = 40;            % initial height (m)

%% --- Ball properties (tennis ball) ---
R = 0.033;           % radius (m)
m = 0.0567;          % mass (kg)
D = 2*R;             % diameter (m)
A = pi*R^2;          % cross-sectional area (m^2)

%% --- Dense time integration until ground (true trajectory) ---
dt_dense = 1e-4;     % small step for "true" solution
Tmax     = 5;        % safety upper bound on time
Nmax     = floor(Tmax/dt_dense) + 1;

t_dense = zeros(Nmax,1);
h_dense = zeros(Nmax,1);
v_dense = zeros(Nmax,1);

t_dense(1) = 0;
h_dense(1) = H0;
v_dense(1) = 0;      % drop from rest (v>0 is downward)

for k = 1:Nmax-1
    t_dense(k+1) = t_dense(k) + dt_dense;

    h = h_dense(k);
    v = v_dense(k);

    % stop when ball hits the ground
    if h <= 0
        t_dense = t_dense(1:k);
        h_dense = h_dense(1:k);
        v_dense = v_dense(1:k);
        break;
    end

    % --- Re-dependent drag model (Brown & Lawler) ---
    Re = rho * abs(v) * D / mu;
    if Re < 1e-12, Re = 1e-12; end

    CD = CD_from_Re_BL(Re);

    % dv/dt (v downward-positive, drag opposes motion)
    %a = g;
    a = g - (0.5 * rho * CD * A / m) * v * abs(v);

    % Euler update
    v_dense(k+1) = v + a * dt_dense;
    h_dense(k+1) = h - v * dt_dense;   % height decreases when v>0
end

t_ground = t_dense(end);

Re_dense = rho * abs(v_dense) * D / mu;     % vector of Re
CD_dense = CD_from_Re_BL(Re_dense);         % vector of CD

%% --- "Camera" sampling at 15 Hz ---
fps    = 15;
dt_cam = 1/fps;

t_cam  = (0:dt_cam:t_ground).';          % camera timestamps

% Interpolate height and velocity at camera times
h_cam  = interp1(t_dense, h_dense, t_cam, 'pchip');
v_cam  = interp1(t_dense, v_dense, t_cam, 'pchip');

N = numel(t_cam);

%% ---  measurement noise to height ---
noise_level = 0.03;                      % ~ 3 cm noise
h_noisy     = h_cam + noise_level * randn(size(h_cam));

%% --- Smooth height and compute derivatives ---
t  = t_cam(:);
dt = dt_cam;

% S-G smoothing, adapt window to signal length
win = min(35, 2*floor((N-1)/2)+1);       % odd, <= N
try
    h_smooth = sgolayfilt(h_noisy, 3, win);
catch
    h_smooth = h_noisy;
end

%%%  sign convention:
%%% v > 0 downward, a > 0 downward.
%%% v = - dh/dt, a = - d2h/dt2

% First derivative: velocity (downward-positive)
v_est = zeros(N,1);
v_est(2:end-1) = -(h_smooth(3:end) - h_smooth(1:end-2)) / (2*dt);
v_est(1)       = v_est(2);
v_est(end)     = v_est(end-1);

% Second derivative: acceleration (downward-positive)
a_est = zeros(N,1);
a_est(2:end-1) = -(h_smooth(3:end) - 2*h_smooth(2:end-1) + h_smooth(1:end-2)) / dt^2;
a_est(1)       = a_est(2);
a_est(end)     = a_est(end-1);

% Signals for SINDy
h   = h_smooth;
v   = v_est;
acc = a_est;
N_true = length(t);

Re_k = rho * abs(v) * D / mu;     % vector of Re
CD_k = CD_from_Re_BL(Re_k);         % vector of CD

% --- 1) Constant acceleration: a = a0 (fit from data, should be ~ g) ---
Phi_const  = ones(N_true,1);
beta_const = Phi_const \ acc;
a0_const   = beta_const;   % single scalar

% --- 2) Linear drag: a ≈ beta0 + beta1 * v ---
Phi_lin  = [ones(N_true,1), v];
beta_lin = Phi_lin \ acc;
a0_lin   = beta_lin(1);
b1_lin   = beta_lin(2);    % a = a0_lin + b1_lin * v

% --- 3) Quadratic drag: a ≈ beta0 + beta2 * v|v| ---
Phi_quad  = [ones(N_true,1), v .* abs(v)];
beta_quad = Phi_quad \ acc;
a0_quad   = beta_quad(1);
b2_quad   = beta_quad(2);  % a = a0_quad + b2_quad * v|v|

fprintf('\nFitted model-based accelerations (downward-positive):\n');
fprintf('  Constant:  a = %.4f\n', a0_const);
fprintf('  Linear:    a = %.4f + (%.4f) v\n', a0_lin,  b1_lin);
fprintf('  Quadratic: a = %.4f + (%.4f) v|v|\n', a0_quad, b2_quad);

% --- Simulate each model on the same time grid ---
h_const = zeros(N_true,1); v_const = zeros(N_true,1);
h_lin   = zeros(N_true,1); v_lin   = zeros(N_true,1);
h_quad  = zeros(N_true,1); v_quad  = zeros(N_true,1);

h_const(1) = H0;  v_const(1) = 0;
h_lin(1)   = H0;  v_lin(1)   = 0;
h_quad(1)  = H0;  v_quad(1)  = 0;

for k = 1:N_true-1
    % Constant-acceleration model
    a_c = a0_const;
    v_const(k+1) = v_const(k) + a_c * dt_cam;
    h_const(k+1) = h_const(k) - v_const(k) * dt_cam;

    % Linear-drag model
    a_l = a0_lin + b1_lin * v_lin(k);
    v_lin(k+1) = v_lin(k) + a_l * dt_cam;
    h_lin(k+1) = h_lin(k) - v_lin(k) * dt_cam;

    % Quadratic-drag model
    a_q = a0_quad + b2_quad * v_quad(k) * abs(v_quad(k));
    v_quad(k+1) = v_quad(k) + a_q * dt_cam;
    h_quad(k+1) = h_quad(k) - v_quad(k) * dt_cam;
end

% --- RMSE vs. smoothed "true" trajectory (height & velocity) ---
rmse_h_const = sqrt(mean((h_const - h).^2));
rmse_h_lin   = sqrt(mean((h_lin   - h).^2));
rmse_h_quad  = sqrt(mean((h_quad  - h).^2));

rmse_v_const = sqrt(mean((v_const - v).^2));
rmse_v_lin   = sqrt(mean((v_lin   - v).^2));
rmse_v_quad  = sqrt(mean((v_quad  - v).^2));

fprintf('\nRMSE (height):   const = %.4f,   linear = %.4f,   quadratic = %.4f\n', ...
        rmse_h_const, rmse_h_lin, rmse_h_quad);
fprintf('RMSE (velocity): const = %.4f,   linear = %.4f,   quadratic = %.4f\n', ...
        rmse_v_const, rmse_v_lin, rmse_v_quad);

% --- Plots: true vs model-based trajectories ---
figure('Name','Height: True vs Model-based','Color','w');
plot(t, h, 'k', 'LineWidth', 2); hold on;
plot(t, h_const, '--', 'LineWidth', 1.2);
plot(t, h_lin,   '-.', 'LineWidth', 1.2);
plot(t, h_quad,  ':',  'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Height (m)');
legend('True (Re-dependent)','Const accel','Linear drag','Quadratic drag', ...
       'Location','best');
title('Height: true vs. simplified model-based trajectories');

figure('Name','Velocity: True vs Model-based','Color','w');
plot(t, v, 'k', 'LineWidth', 2); hold on;
plot(t, v_const, '--', 'LineWidth', 1.2);
plot(t, v_lin,   '-.', 'LineWidth', 1.2);
plot(t, v_quad,  ':',  'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s, down +)');
legend('True (Re-dependent)','Const accel','Linear drag','Quadratic drag', ...
       'Location','best');
title('Velocity: true vs. simplified model-based trajectories');


%% ==== SINDy regression: library in h and v ====
t   = t(:);
h   = h(:);
v   = v(:);
acc = acc(:);

N = length(t);

names = {
    '1', ...
    'h', ...
    'v', ...
    'h^2',...
    'h*v', ...
    'v^2', ...
    'h^3',...
    'h^2*v',...
    'h*v^2',...
    'v^3' ...
};

phi1  = ones(N,1);
phi2  = h;
phi3  = v;
phi4  = h.^2;
phi5  = h.*v;
phi6  = v.^2;
phi7  = h.^3;
phi8  = (h.^2).*v;
phi9  = h.*(v.^2);
phi10 = v.^3;

Theta = [phi1 phi2 phi3 phi4 phi5 phi6 phi7 phi8 phi9 phi10];

P = size(Theta,2);

Col_norm = sqrt(sum(Theta.^2,1)); % Compute norms of columns
Col_norm(Col_norm == 0) = 1;
Theta_norm = Theta ./ Col_norm;

% Initial least-squares fit
Ei = Theta_norm \ acc;   % P×1

% Sequential thresholded least squares
delta = 0.08;      % sparsity threshold 
max_iter = 10;

for it = 1:max_iter
    small_idx = abs(Ei) < delta;
    big_idx   = ~small_idx;

    if all(~big_idx)
        % everything got zeroed; break and keep all-zero
        break;
    end

    % zero out small coefficients
    Ei(small_idx) = 0;

    % refit only on the "big" terms
    Theta_big = Theta(:, big_idx);
    Ei_big    = Theta_big \ acc;
    Ei(big_idx) = Ei_big;
end

E = Ei ./ Col_norm.';

%%% contributions using original Theta and Xi
contrib = Theta .* (E.');    % N×P, each column = term(t) * coeff
ci      = vecnorm(contrib, 2, 1);   % 1×P, L2 over time

%% === Print ranked contributions ===
fprintf('\nOne-ball realistic experiment contributions (STLSQ SINDy):\n');
[ci_sorted, idx] = sort(ci,'descend');
for i = 1:P
    fprintf('%10s : %.6f   (E = %+ .6f)\n', names{idx(i)}, ci_sorted(i), E(idx(i)));
end

figure('Name','C_D vs Re (Brown-Lawler) at t_cam');
loglog(Re_k, CD_k, '-o', 'LineWidth', 2, ...
       'MarkerSize', 4, 'MarkerFaceColor', 'white');
grid on;
xlabel('Reynolds number, Re');
ylabel('Drag coefficient, C_D');
title('C_D vs Re (Brown-Lawler, simulated trajectory) ; tdense');

%% === Plot: dominant candidate functions ===
figure('Name','Dominant Candidate Functions ','Color','w');
b = bar(ci, 'LineWidth', 1.5);
b.FaceColor = [0.20 0.60 0.70];   % RGB color (nice muted blue)
b.EdgeColor = 'none';
grid on;
set(gca,'XTick',1:P,'XTickLabel',names, ...
    'XTickLabelRotation',45,'FontSize',12);
ylabel('Contribution (L2 norm)');
title(sprintf('Candidate Function Contributions (One Ball, \\delta = %.3f, Noise = %.3f)', delta, noise_level));


%% === Quick sanity check: h(t), v(t), a(t) ===
figure('Name','Trajectories','Color','w');
subplot(3,1,1);
plot(t,h,'LineWidth',1.5); grid on;
ylabel('h (m)');
title('Height');

subplot(3,1,2);
plot(t,v,'LineWidth',1.5); grid on;
ylabel('v (m/s)');
title('Velocity (down +)');

subplot(3,1,3);
plot(t,acc,'LineWidth',1.5); grid on;
ylabel('a (m/s^2)');
xlabel('t (s)');
title('Acceleration (from differentiation, down +)');


%% ================= Local function: C_D(Re) =====================
function CD = CD_from_Re_BL(Re)
    % Brown & Lawler (2003) correlation used in the paper
    % CD(Re) = 24/Re * (1 + 0.150 Re^0.681) + 0.407 / (1 + 8710/Re)
    Re(Re < 1e-6) = 1e-6;
    CD = 24./Re .* (1 + 0.150 * Re.^0.681) + 0.407 ./ (1 + 8710./Re);
end
