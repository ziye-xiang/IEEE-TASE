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

function metrics = QMIX_evaluate_core(agent,env_step_fn,action_decoder_fn, ...
    demand_provider_fn,X0,cfg)
%QMIX_EVALUATE_CORE Greedy evaluation of the trained QMIX policy.

state = X0;
ctrl_state = struct();
reward_sum = 0;
resource_abs_error = 0;
resource_count = 0;

for step = 1:cfg.max_steps
    state(:,cfg.state_dim) = demand_provider_fn(1,step,cfg.agent_num);
    local_obs = double(state)' ./ double(cfg.obs_scale(:));

    joint_actions = agent.select_actions(local_obs,0);
    action_cell = action_decoder_fn(joint_actions);

    [next_state,r_team,ctrl_state,info] = ...
        env_step_fn(state,action_cell,ctrl_state,cfg);

    reward_sum = reward_sum + r_team;
    resource_abs_error = resource_abs_error + ...
        sum(abs(info.U_required-info.U_actual));
    resource_count = resource_count + cfg.agent_num;
    state = next_state;
end

metrics.team_reward = reward_sum;
metrics.resource_mae = resource_abs_error/max(resource_count,1);
end
