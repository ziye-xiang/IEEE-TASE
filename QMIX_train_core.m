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

function results = QMIX_train_core(env_step_fn, action_decoder_fn, demand_provider_fn, X0, cfg)
%QMIX_TRAIN_CORE Core training procedure for QMIX-based TDM allocation.
%
% Required user callbacks:
%   env_step_fn(state, action_cell, ctrl_state, cfg)
%   action_decoder_fn(joint_actions)
%   demand_provider_fn(episode, step, agent_num)
%
% The callback env_step_fn must return:
%   [next_state, team_reward, next_ctrl_state]

arguments
    env_step_fn (1,1) function_handle
    action_decoder_fn (1,1) function_handle
    demand_provider_fn (1,1) function_handle
    X0 double
    cfg struct
end

required = {'obs_dim','state_dim','action_dim','agent_num','gamma', ...
    'learning_rate','learning_rate_final','batch_size','memory_size', ...
    'num_episodes','max_steps','learning_starts','target_update_interval', ...
    'epsilon_start','epsilon_min','epsilon_decay_steps','obs_scale'};
for k = 1:numel(required)
    assert(isfield(cfg,required{k}), 'Missing cfg.%s', required{k});
end

agent = QMIX_Agent(cfg.obs_dim, cfg.obs_dim*cfg.agent_num, ...
    cfg.action_dim, cfg.agent_num, cfg.gamma, cfg.learning_rate, ...
    cfg.memory_size, cfg.batch_size);

episode_reward = zeros(cfg.num_episodes,1);
episode_loss   = nan(cfg.num_episodes,1);
epsilon_hist   = zeros(cfg.num_episodes,1);
lr_hist        = zeros(cfg.num_episodes,1);

total_steps = 0;

for ep = 1:cfg.num_episodes
    % ----- learning-rate schedule -----
    p_lr = min(max((ep-1)/max(cfg.num_episodes-1,1),0),1);
    lr = cfg.learning_rate_final + ...
        0.5*(cfg.learning_rate-cfg.learning_rate_final)*(1+cos(pi*p_lr));
    agent.learning_rate = lr;
    lr_hist(ep) = lr;

    state = X0;
    ctrl_state = struct();
    reward_sum = 0;
    loss_sum = 0;
    loss_count = 0;

    for step = 1:cfg.max_steps
        % Update the resource demand for the current decision step.
        state(:,cfg.state_dim) = demand_provider_fn(ep,step,cfg.agent_num);

        local_obs = encode_local_obs(state,cfg.obs_scale);
        global_state = reshape(local_obs,[],1);

        % ----- epsilon-greedy decentralized action selection -----
        epsilon = cfg.epsilon_min + ...
            (cfg.epsilon_start-cfg.epsilon_min)* ...
            max(1-total_steps/max(cfg.epsilon_decay_steps,1),0);
        joint_actions = agent.select_actions(local_obs,epsilon);

        % Convert discrete actions to TDM allocation patterns.
        action_cell = action_decoder_fn(joint_actions);

        % ----- closed-loop environment -----
        [next_state,team_reward,next_ctrl_state] = ...
            env_step_fn(state,action_cell,ctrl_state,cfg);

        next_local_obs = encode_local_obs(next_state,cfg.obs_scale);
        next_global_state = reshape(next_local_obs,[],1);
        done = (step == cfg.max_steps);

        % ----- replay and QMIX update -----
        agent.store_experience(local_obs,global_state,joint_actions, ...
            team_reward,next_local_obs,next_global_state,done);

        if agent.memory_counter >= cfg.learning_starts
            loss_now = agent.train_step();
            loss_sum = loss_sum + loss_now;
            loss_count = loss_count + 1;
        end

        state = next_state;
        ctrl_state = next_ctrl_state;
        reward_sum = reward_sum + team_reward;
        total_steps = total_steps + 1;

        if done
            break;
        end
    end

    if mod(ep,cfg.target_update_interval)==0 && ...
            agent.memory_counter >= cfg.learning_starts
        agent.hard_update_targets();
    end

    episode_reward(ep) = reward_sum;
    if loss_count>0
        episode_loss(ep) = loss_sum/loss_count;
    end
    epsilon_hist(ep) = epsilon;
end

results.agent = agent;
results.episode_reward = episode_reward;
results.episode_loss = episode_loss;
results.epsilon = epsilon_hist;
results.learning_rate = lr_hist;
end

function obs = encode_local_obs(state,scale)
% Normalize the local observations.
obs = double(state)' ./ double(scale(:));
end
