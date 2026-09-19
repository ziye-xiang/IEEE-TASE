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

classdef QMIX_Agent < handle
    %QMIX_AGENT Core QMIX agent implementation.
    %
    % Decentralized execution:
    %   Each agent selects its action using only its local observation.
    %
    % Centralized training:
    %   Individual Q_i values are combined by a monotonic QMIX mixer into Q_tot,
    %   and temporal-difference learning uses the shared team reward.
    %
    % This implementation:
    %   1) uses one 64-64 local Q-network per agent;
    %   2) all admissible discrete actions participate in epsilon-greedy;
    %   3) uses no dual branch, resource pretraining, policy freezing, or guided exploration;
    %   4) Double-Q target selects the next action over the admissible action set.

    properties
        obs_dim
        global_state_dim
        action_dim
        num_agents
        gamma
        learning_rate
        memory_capacity
        batch_size
        grad_clip

        agent_nets
        target_agent_nets
        mixer
        target_mixer

        memory_buffer
        memory_counter
        memory_ptr

        update_count
        total_steps
        last_td_loss
    end

    methods
        function obj = QMIX_Agent(obs_dim, global_state_dim, action_dim, num_agents, ...
                                  gamma, learning_rate, memory_capacity, batch_size)
            obj.obs_dim = obs_dim;
            obj.global_state_dim = global_state_dim;
            obj.action_dim = action_dim;
            obj.num_agents = num_agents;
            obj.gamma = gamma;
            obj.learning_rate = learning_rate;
            obj.memory_capacity = memory_capacity;
            obj.batch_size = batch_size;
            obj.grad_clip = 10;

            obj.memory_counter = 0;
            obj.memory_ptr = 1;
            obj.update_count = 0;
            obj.total_steps = 0;
            obj.last_td_loss = NaN;

            fprintf(['Initializing standard QMIX: local observation dim=%d, global state dim=%d, ' ...
                     'action dim=%d, number of agents=%d\n'], ...
                    obs_dim, global_state_dim, action_dim, num_agents);

            obj.agent_nets = cell(num_agents,1);
            obj.target_agent_nets = cell(num_agents,1);
            for i = 1:num_agents
                obj.agent_nets{i} = QMIXMLP(obs_dim,[64,64],action_dim);
                % Reduce the initial Q-value scale to avoid excessively large initial Q_tot values.
                obj.agent_nets{i}.layers{end}.W = ...
                    0.1*obj.agent_nets{i}.layers{end}.W;

                obj.target_agent_nets{i} = QMIXMLP(obs_dim,[64,64],action_dim);
                obj.target_agent_nets{i}.copy_weights(obj.agent_nets{i});
            end

            obj.mixer = QMIXMixer(num_agents,global_state_dim,32,64);
            obj.mixer.hyper_w1.layers{end}.W = ...
                0.1*obj.mixer.hyper_w1.layers{end}.W;
            obj.mixer.hyper_w2.layers{end}.W = ...
                0.1*obj.mixer.hyper_w2.layers{end}.W;

            obj.target_mixer = QMIXMixer(num_agents,global_state_dim,32,64);
            obj.target_mixer.copy_weights(obj.mixer);

            obj.memory_buffer = repmat(struct( ...
                'local_obs',zeros(obs_dim,num_agents,'single'), ...
                'global_state',zeros(global_state_dim,1,'single'), ...
                'joint_actions',zeros(num_agents,1,'single'), ...
                'team_reward',single(0), ...
                'next_local_obs',zeros(obs_dim,num_agents,'single'), ...
                'next_global_state',zeros(global_state_dim,1,'single'), ...
                'done',false),memory_capacity,1);
        end

        function joint_actions = select_actions(obj,local_obs,epsilon)
            % Standard epsilon-greedy over all admissible actions.
            local_obs = obj.validate_local_obs(local_obs);
            epsilon = min(max(double(epsilon),0),1);

            joint_actions = zeros(obj.num_agents,1);
            for i = 1:obj.num_agents
                if rand <= epsilon
                    joint_actions(i) = randi(obj.action_dim);
                else
                    q = obj.agent_nets{i}.forward(local_obs(:,i));
                    [~,joint_actions(i)] = max(q);
                end
            end
        end

        function store_experience(obj,local_obs,global_state,joint_actions,team_reward, ...
                                  next_local_obs,next_global_state,done)
            idx = obj.memory_ptr;
            obj.memory_ptr = mod(obj.memory_ptr,obj.memory_capacity)+1;
            obj.memory_counter = min(obj.memory_counter+1,obj.memory_capacity);

            local_obs = obj.validate_local_obs(local_obs);
            next_local_obs = obj.validate_local_obs(next_local_obs);
            global_state = obj.adjust_vector(global_state,obj.global_state_dim);
            next_global_state = obj.adjust_vector(next_global_state,obj.global_state_dim);
            joint_actions = round(obj.adjust_vector(joint_actions,obj.num_agents));
            joint_actions = max(1,min(joint_actions,obj.action_dim));

            team_reward = double(team_reward);
            if numel(team_reward) ~= 1
                team_reward = sum(team_reward(:));
            end
            if ~isfinite(team_reward)
                error('store_experience: team_reward must be a finite scalar.');
            end

            obj.memory_buffer(idx).local_obs = single(local_obs);
            obj.memory_buffer(idx).global_state = single(global_state);
            obj.memory_buffer(idx).joint_actions = single(joint_actions);
            obj.memory_buffer(idx).team_reward = single(team_reward);
            obj.memory_buffer(idx).next_local_obs = single(next_local_obs);
            obj.memory_buffer(idx).next_global_state = single(next_global_state);
            obj.memory_buffer(idx).done = logical(done);
        end

        function loss = train_step(obj)
            if obj.memory_counter < obj.batch_size
                loss = 0;
                return;
            end

            batch = obj.sample_batch();
            B = size(batch.global_states,2);

            agent_grad_sum = cell(obj.num_agents,1);
            for i = 1:obj.num_agents
                agent_grad_sum{i} = obj.agent_nets{i}.zero_grads();
            end
            mixer_grad_sum = obj.mixer.zero_grads();

            total_loss = 0;
            td_abs_sum = 0;

            for k = 1:B
                q_taken = zeros(obj.num_agents,1);
                agent_cache = cell(obj.num_agents,1);
                actions_k = round(batch.joint_actions(:,k));

                % ---------- Current Q_tot ----------
                for i = 1:obj.num_agents
                    [q_all,agent_cache{i}] = ...
                        obj.agent_nets{i}.forward_cache(batch.local_obs(:,i,k));
                    ai = max(1,min(actions_k(i),obj.action_dim));
                    q_taken(i) = q_all(ai);
                end

                [q_tot,mix_cache] = ...
                    obj.mixer.forward_cache(q_taken,batch.global_states(:,k));

                % ---------- Double-Q target ----------
                target_q_agents = zeros(obj.num_agents,1);
                for i = 1:obj.num_agents
                    next_obs_i = batch.next_local_obs(:,i,k);

                    q_online_next = obj.agent_nets{i}.forward(next_obs_i);
                    [~,greedy_next_action] = max(q_online_next);

                    q_target_next = obj.target_agent_nets{i}.forward(next_obs_i);
                    target_q_agents(i) = q_target_next(greedy_next_action);
                end

                q_tot_target_next = obj.target_mixer.forward( ...
                    target_q_agents,batch.next_global_states(:,k));

                y = batch.rewards(k) + obj.gamma* ...
                    (1-double(batch.dones(k)))*q_tot_target_next;

                td = q_tot-y;
                total_loss = total_loss+td^2;
                td_abs_sum = td_abs_sum+abs(td);

                [grad_q_agents,mix_grads] = obj.mixer.backward(mix_cache,2*td);

                % Accumulate mixer gradients layer by layer.
                mixer_grad_sum = obj.mixer.add_grads(mixer_grad_sum,mix_grads);

                % Accumulate local-agent Q-network gradients layer by layer.
                for i = 1:obj.num_agents
                    ai = max(1,min(actions_k(i),obj.action_dim));
                    grad_q_vector = zeros(obj.action_dim,1);
                    grad_q_vector(ai) = grad_q_agents(i);

                    [~,g_agent] = obj.agent_nets{i}.backward( ...
                        agent_cache{i},grad_q_vector);

                    for ll = 1:numel(agent_grad_sum{i})
                        agent_grad_sum{i}{ll}.W = ...
                            agent_grad_sum{i}{ll}.W + g_agent{ll}.W;
                        agent_grad_sum{i}{ll}.b = ...
                            agent_grad_sum{i}{ll}.b + g_agent{ll}.b;
                    end
                end
            end

            scale = 1/B;
            for i = 1:obj.num_agents
                obj.agent_nets{i}.apply_gradients( ...
                    agent_grad_sum{i},obj.learning_rate,scale,obj.grad_clip);
            end
            obj.mixer.apply_gradients( ...
                mixer_grad_sum,obj.learning_rate,scale,obj.grad_clip);

            obj.update_count = obj.update_count+1;
            obj.total_steps = obj.total_steps+1;
            loss = total_loss/B;
            obj.last_td_loss = loss;

            if ~isfinite(loss)
                warning('QMIX produced a non-finite loss. mean|TD|=%.6g', ...
                    td_abs_sum/max(1,B));
            end
        end

        function hard_update_targets(obj)
            for i = 1:obj.num_agents
                obj.target_agent_nets{i}.copy_weights(obj.agent_nets{i});
            end
            obj.target_mixer.copy_weights(obj.mixer);
        end

        function batch = sample_batch(obj)
            current_size = min(obj.memory_counter,obj.memory_capacity);
            B = min(obj.batch_size,current_size);
            indices = randperm(current_size,B);

            batch.local_obs = zeros(obj.obs_dim,obj.num_agents,B);
            batch.global_states = zeros(obj.global_state_dim,B);
            batch.joint_actions = zeros(obj.num_agents,B);
            batch.rewards = zeros(1,B);
            batch.next_local_obs = zeros(obj.obs_dim,obj.num_agents,B);
            batch.next_global_states = zeros(obj.global_state_dim,B);
            batch.dones = false(1,B);

            for k = 1:B
                e = obj.memory_buffer(indices(k));
                batch.local_obs(:,:,k) = double(e.local_obs);
                batch.global_states(:,k) = double(e.global_state);
                batch.joint_actions(:,k) = double(e.joint_actions);
                batch.rewards(k) = double(e.team_reward);
                batch.next_local_obs(:,:,k) = double(e.next_local_obs);
                batch.next_global_states(:,k) = double(e.next_global_state);
                batch.dones(k) = logical(e.done);
            end
        end

        function snapshot = get_policy_snapshot(obj)
            snapshot.agent = cell(obj.num_agents,1);
            for i = 1:obj.num_agents
                snapshot.agent{i} = obj.agent_nets{i}.get_snapshot();
            end
            snapshot.mixer = obj.mixer.get_snapshot();
        end

        function restore_policy_snapshot(obj,snapshot)
            if isempty(snapshot), return; end

            for i = 1:obj.num_agents
                obj.agent_nets{i}.restore_snapshot(snapshot.agent{i});
                obj.agent_nets{i}.reset_optimizer();
            end
            obj.mixer.restore_snapshot(snapshot.mixer);
            obj.mixer.reset_optimizer();
            obj.hard_update_targets();
        end

        function save_model(obj,filepath)
            model_data = obj.get_policy_snapshot();
            save(filepath,'model_data','-v7.3');
            fprintf('QMIX model saved to: %s\n',filepath);
        end
    end

    methods (Access=private)
        function obs = validate_local_obs(obj,obs)
            obs = double(obs);
            if isequal(size(obs),[obj.obs_dim,obj.num_agents])
                return;
            end
            if numel(obs) ~= obj.obs_dim*obj.num_agents
                error('Local observation dimension mismatch: expected %d-by-%d, received %s.', ...
                    obj.obs_dim,obj.num_agents,mat2str(size(obs)));
            end
            obs = reshape(obs,obj.obs_dim,obj.num_agents);
        end

        function v = adjust_vector(~,v,target_dim)
            v = double(v(:));
            if numel(v)>target_dim
                v = v(1:target_dim);
            elseif numel(v)<target_dim
                v = [v;zeros(target_dim-numel(v),1)];
            end
        end
    end
end
