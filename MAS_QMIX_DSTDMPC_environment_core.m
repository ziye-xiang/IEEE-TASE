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

function [next_state,r_team,next_ctrl_state,info] = ...
    MAS_QMIX_DSTDMPC_environment_core(state,action_cell,ctrl_state,cfg)
%MAS_QMIX_DSTDMPC_ENVIRONMENT_CORE Core environment interface for QMIX, TDM, and Dual ST-DMPC.
%
% The function couples QMIX resource-allocation actions with local Dual
% ST-DMPC controller updates and evaluates the normalized team reward.
% The local controller update is supplied through cfg.controller_step_fn.

arguments
    state double
    action_cell cell
    ctrl_state struct
    cfg struct
end

required = {'agent_num','controller_step_fn','resource_util_factor', ...
    'Fi','Gi','delta_U','P_reward','J_ref'};
for k = 1:numel(required)
    assert(isfield(cfg,required{k}), 'Missing cfg.%s', required{k});
end

nA = cfg.agent_num;
next_state = state;
next_ctrl_state = ctrl_state;
terminal_state = zeros(nA,3);

% -------------------------------------------------------------------------
% 1) Apply each agent's TDM allocation to its local Dual ST-DMPC controller
% -------------------------------------------------------------------------
for i = 1:nA
    % The controller callback encapsulates self-trigger prediction,
    % active/inactive-slot handling, MPC solution and input application.
    [x_next,next_ctrl_state] = cfg.controller_step_fn( ...
        i,state(i,1:3),action_cell{i},next_ctrl_state,cfg);

    terminal_state(i,:) = x_next(:)';
    next_state(i,1:3) = terminal_state(i,:);
end

% -------------------------------------------------------------------------
% 2) Normalized two-term team reward
% -------------------------------------------------------------------------
U_required = state(:,end)';
U_actual = zeros(1,nA);
for i = 1:nA
    a = double(action_cell{i}(:));
    U_actual(i) = mean(a)*cfg.resource_util_factor;
end

resource_error_norm = min(abs(U_required-U_actual)./cfg.delta_U,1);
resource_score = 1-resource_error_norm;

J = zeros(1,nA);
for i = 1:nA
    xi = terminal_state(i,:);
    J(i) = xi*cfg.P_reward*xi';
end
control_score = 1-min(J./max(cfg.J_ref,1e-12),1);

r_individual = cfg.Fi.*resource_score + cfg.Gi.*control_score;
r_team = sum(r_individual);

info.U_required = U_required;
info.U_actual = U_actual;
info.resource_score = resource_score;
info.control_score = control_score;
info.r_individual = r_individual;
end
