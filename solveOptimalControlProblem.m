% =========================================================================
% Copyright (c) 2026 The Authors. All rights reserved.
%
% Associated manuscript:
% IEEE Transactions on Automation Science and Engineering (IEEE T-ASE)
% Manuscript ID: T-ASE-2026-2601
%
% This code provides the core implementation of the proposed method and is
% intended for academic research purposes.
% Redistribution or commercial use requires permission from the authors.
% =========================================================================

function [u,V,exitflag,output] = solveOptimalControlProblem( ...
    runningcosts,disrunningcosts,terminalcosts,constraints,terminalconstraints, ...
    linearconstraints,system,neigbourI,gamma,N,n,t0,x0,pre_x,index,u0,T, ...
    atol_ode_sim,rtol_ode_sim,tol_opt,options,type) %#ok<INUSD>
%SOLVEOPTIMALCONTROLPROBLEM Solve the local finite-horizon DMPC problem.
%
% The decision variable is the finite-horizon control sequence u. The local
% stage cost, neighbor-coupling cost, terminal cost, linear constraints, and
% nonlinear terminal constraints are assembled through callback functions.

% Evaluate the current warm-start trajectory before optimization.
computeOpenloopSolution(system,N,T,t0,x0,u0,atol_ode_sim,rtol_ode_sim,type);

% Construct linear inequality/equality constraints and input bounds.
[A,b,Aeq,beq,lb,ub] = linearconstraints(N);

% Solve the constrained finite-horizon optimization problem.
[u,V,exitflag,output] = fmincon( ...
    @(uu) costfunction(runningcosts,disrunningcosts,terminalcosts, ...
        system,neigbourI,gamma,N,T,t0,x0,pre_x,index,uu, ...
        atol_ode_sim,rtol_ode_sim,type), ...
    u0,A,b,Aeq,beq,lb,ub, ...
    @(uu) nonlinearconstraints(constraints,terminalconstraints, ...
        system,N,n,T,t0,x0,uu,atol_ode_sim,rtol_ode_sim,type), ...
    options);
end
