
function [X_vector,U_vector]=nmpc2(refs)


% Data
m = 0.8;                                                                   % Mass of Quadrotor
L = 0.165;                                                                  % Moment arm length
k_t = 3 * 10^(-6);                                                         % Thrust constant
b = 1 * 10^(-7);                                                           % Moment constant
g = 9.81;                                                                  % Acceleration due to gravity
k_d = 0.25;                                                                % Drag constant
Ixx = 5 * 10^(-3);                                                         % Momentds of inertia
Iyy = 5 * 10^(-3);
Izz = 1 * 10^(-2);
c_m = 1 * 10^(4);                                                          % Moment constant
Tmax = 50;                                                                 % Maximum Time duration

% System Sizes
nx = 12;                                                                   % Nb of states
nu = 4;                                                                    % Nb of inputs
ny = 6;                                                                    % Nb of outputs

x0_quadcopter = zeros(nx,1);

% Inputs at equilbrium point at all states=0
u_eq = g*m/(4*k_t*c_m);

N = 15;                                                                    % prediction horizon
umax = 100-u_eq; umin = u_eq;                                              % input limit
zmax = inf; zmin = inf;                                                    % room limits 
xymax = inf;

%%
% Construction of the linear state space model

% symbolic variables
syms x y z v_x v_y v_z phi theta psi1 p q r u1 u2 u3 u4
% state vector
state = [x; y; z; v_x; v_y; v_z; phi; theta; psi1; p; q; r];
% input vector
input = [u1; u2; u3; u4];

% The functions f:
f1 = v_x;
f2 = v_y;
f3 = v_z;

f4 = -k_d/m * v_x + k_t*c_m/m *(sin(psi1)*sin(phi)+cos(psi1)*cos(phi)*sin(theta))*(u1+u2+u3+u4);
f5 = -k_d/m * v_y + k_t*c_m/m *(cos(phi)*sin(psi1)*sin(theta)-cos(psi1)*sin(phi))*(u1+u2+u3+u4);
f6 = -k_d/m * v_z - g + k_t*c_m/m *(cos(theta)*cos(phi))*(u1+u2+u3+u4);

f7 = p + q*(sin(phi)*tan(theta)) + r*(cos(phi)*tan(theta));
f8 = q*cos(phi) - r*sin(phi);
f9 = sin(phi)/cos(theta) *q + cos(phi)/cos(theta) * r;

f10 = L*k_t*c_m/Ixx * (u1-u3) - (Iyy-Izz)/Ixx * q*r;
f11 = L*k_t*c_m/Iyy * (u2-u4) - (Izz-Ixx)/Iyy * p*r;
f12 = b*c_m/Izz * (u1-u2+u3-u4) - (Ixx-Iyy)/Izz * q*p;

F = [f1; f2; f3; f4; f5; f6; f7; f8; f9; f10; f11; f12];

%%
% Deriving the functions in the state variables (Jacobian)
J = jacobian(F, state);
 
% Evaluating the jacobian in the equilibrium values: the result is A
A = subs(J,[state; input],[zeros(nx,1); u_eq*ones(nu,1)]);
A = double(A);


% Deriving the functions in the input variables
J = jacobian(F, input);

% Evaluating the derivatives in the equilibrium values: the result is B
B = subs(J, [state; input],[zeros(nx,1); u_eq*ones(nu,1) ]);
B = double(B);

% The output consists of states 1 to 3 and 7 to 9, so C selects these and D
% is zero
C = [eye(3), zeros(3,9);
     zeros(3,6), eye(3), zeros(3,3)];
D = zeros(ny,nu);

% Creating the continuous time system
c_sys = ss(A,B,C,D);


%Checking the stability
disp('Poles(continous system):')
disp(eig(A))

%%
%Discretization
T_s = 0.9; % Sampling time

M = inv(eye(nx) - A*T_s/2)


[A_d,B_d,C_d,D_d]=c2dm(A,B,C,D,T_s,'zoh');                                 %Discretization using zero order hold

% Creating the discrete time system
sys = ss(A_d,B_d,C_d,D_d,T_s);

%%
%Checking the stability

disp('Poles(discretized system):')
disp(eig (A_d))



disp ('Controllability matrix');

CO = ctrb(A_d,B_d);

disp('Rank of the controllability matrix:');
rank(CO)



disp ('Observability matrix');

OB = obsv(A_d,C_d);

disp('Rank of the observability matrix:');
rank(OB)

%% Setpoints


[M,~] = size(refs);
y_ref_vector = reshape(refs',[M*ny,1]);                                    % Built optimization problem that finds u such that (y-y_ref) minimized


F = zeros(M*ny,nx);                                                        % Establish O
F(1:ny,:) = C_d;
for i=1:M-1
    F(i*ny+1:(i+1)*ny,:) = F((i-1)*ny+1:i*ny,:)*A_d;
end


H_o = zeros(M*ny,nu);                                                     % Establish H: H_0 = D, H_k = CA^{k-1}B
H_o(1:ny,:)=D_d;
temp = B_d;
for k=1:M-1
    H_o(k*ny+1:(k+1)*ny,:)=C_d*temp;
    temp = A_d*temp;
end
H = zeros(M*ny,M*nu);
for i=0:M-1
    H(:,i*nu+1:(i+1)*nu) = H_o;
    H_o = [zeros(ny, nu); 
            H_o(1:end-ny,:)];
end


Q = diag(repmat([0.001,0.001,0.001,0,0,0],1,M));
R = 1.0e-4*eye(M*nu);
u_ref_vector = quadprog(R+H'*Q*H,H'*Q'*(y_ref_vector-F*x0_quadcopter));
u_ref_vector=-u_ref_vector;


u_ref = reshape(u_ref_vector,[nu,M])';                                     % Reshape the inputs as rowvectors for each time t

[Y, T, x_ref] = lsim(sys,u_ref);                                           % Simulation to get actual outputs and reference states

x_ref_vector = reshape(x_ref',[M*nx,1]);

close all



%%
for i=(M+1):(M+N)                                                          % Reference vector with N repititions of the last input/state  
    x_ref_vector((i-1)*nx+1:i*nx) = x_ref_vector(i-nx+1:i);
    u_ref_vector((i-1)*nu+1:i*nu) = u_ref_vector(i-nu+1:i);
end

qdiag = ones(1,nx);
rdiag = 1e-7*ones(1,nu);
bigQ = repmat(qdiag,1,N);
bigR = repmat(rdiag,1,N);
H = 2*diag([bigQ,bigR]);
A_eq = [eye(nx*N), zeros(nx*N,nu*N)];
for i=1:N
    if(i<N)
        A_eq(i*nx+1:(i+1)*nx,(i-1)*nx+1:i*nx) = -A_d;
    end
    A_eq((i-1)*nx+1:i*nx, nx*N + (i-1)*nu+1:nx*N + i*nu) = -B_d;
end

A_limit = [eye(nx*N), zeros(nx*N,nu*N); -eye(nx*N), zeros(nx*N,nu*N);      % Limiting the space of the quadcopter and the input size
zeros(nu*N,nx*N), eye(nu*N); zeros(nu*N,nx*N), -eye(nu*N)]; 
b_limit_u_max = repmat([umax],nu,1);                                       % Inputs between 0 and umax
b_limit_u_min = repmat([umin],nu,1);
b_limit_x_max = [xymax; xymax; zmax; inf*ones(nx-3,1)];                    % XY coordinates between + and - xymax, z coordinate between 0 and zmax
b_limit_x_min = [xymax; xymax; zmin; inf*ones(nx-3,1)];
b_limit = [repmat(b_limit_x_max,N,1); repmat(b_limit_x_min,N,1) ;repmat(b_limit_u_max,N,1); repmat(b_limit_u_min,N,1)];

x = x0_quadcopter; 
for k=1:M
    
    x_ref = x_ref_vector((k-1)*nx+1:(k+N-1)*nx);                           % Extract time horizon
    u_ref = u_ref_vector((k-1)*nu+1:(k+N-1)*nu);
    
    f = -H*[x_ref; u_ref];                                                 % Complete optimization matrices
    b_eq = zeros(N*nx,1);
    b_eq(1:nx) = A_d*x;
    
    xu = quadprog(H,f,A_limit,b_limit,A_eq,b_eq);                          % Solve optimization problem
    %xu = quadprog(H,f,[],[],A_eq,b_eq);
    u = xu(nx*N+1:nx*N+nu);                                                % Applying first input
    y = C_d*x + D_d*u;                                                     % Update state
    x = A_d*x + B_d*u;
    U_vector(k,:) = u';
    X_vector(k,:) = x';
    Y_vector(k,:) = y';
end

close all


%%

figure                                                                     % Plotting
plot(T,Y_vector(:,1));
hold on
plot(T,refs(:,1),'-.');
xlabel('T [s]')
legend({'x','x_{ref}'},'FontSize',12);
title('X-axis Response');
grid on

figure
plot(T,Y_vector(:,2));
hold on
plot(T,refs(:,2),'-.');
xlabel('T [s]')
legend({'y','y_{ref}'},'FontSize',12);
title('Y-axis Response');
grid on

figure
plot(T,Y_vector(:,3));
hold on
plot(T,refs(:,3),'-.');
xlabel('T [s]')
legend({'z','z_{ref}'},'FontSize',12);
title('Z-axis Response');
grid on

figure
plot3(refs(:,1),refs(:,2),refs(:,3),'r');
hold on
plot3(Y_vector(:,1),Y_vector(:,2),Y_vector(:,3),'b');
legend({'reference','quadcopter'},'FontSize',12);
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')
title('Simulation results NMPC')
grid on

% figure
% plot(T,X_vector);
% legend({'x','y','z','v_x','v_y','v_z','\phi','\tau','\psi','\omega_x','\omega_y','\omega_z'},'FontSize',12);
% title('States');
% xlabel('T [s]')
% figure
% plot(T,X_vector(:,1));
% legend('x','FontSize',12);
% title('States');
% xlabel('T [s]')
% figure
% plot(T,X_vector(:,2));
% legend('y','FontSize',12);
% title('States');
% xlabel('T [s]')
% figure
% plot(T,X_vector(:,3));
% legend('z','FontSize',12);
% title('States');
% xlabel('T [s]')
figure
plot(T,X_vector(:,4));
legend('v_x','FontSize',12);
title('States');
xlabel('T [s]')
grid on

figure
plot(T,X_vector(:,5));
legend('v_y','FontSize',12);
title('States');
xlabel('T [s]')
grid on

figure
plot(T,X_vector(:,6));
legend('v_z','FontSize',12);
title('States');
xlabel('T [s]')
grid on

figure
plot(T,X_vector(:,7));
legend('\phi','FontSize',12);
title('States');
xlabel('T [s]')
grid on

figure
plot(T,X_vector(:,8));
legend('\theta','FontSize',12);
title('States');
xlabel('T [s]')
grid on

figure
plot(T,X_vector(:,9));
legend('\psi','FontSize',12);
title('States');
xlabel('T [s]')
grid on

% figure
% plot(T,X_vector(:,10));
% legend('\omega_x','FontSize',12);
% title('States');
% xlabel('T [s]')
% figure
% plot(T,X_vector(:,11));
% legend('\omega_y','FontSize',12);
% title('States');
% xlabel('T [s]')
% figure
% plot(T,X_vector(:,12));
% legend('\omega_z','FontSize',12);
% title('States');
% xlabel('T [s]')

figure
plot(T,U_vector(:,1));
title('Control input U1');
xlabel('T [s]')
grid on

figure
plot(T,U_vector(:,2));
title('Control input U2');
xlabel('T [s]')
grid on

figure
plot(T,U_vector(:,3));
title('Control input U3');
xlabel('T [s]')
grid on

figure
plot(T,U_vector(:,4));
title('Control input U4');
xlabel('T [s]')
grid on


%%
end
