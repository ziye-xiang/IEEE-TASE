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

classdef QMIXMixer < handle
    %QMIXMIXER Standard monotonic QMIX mixing network.
    % Qtot=f_mix(Q1,...,Qn;s)
    % The hypernetworks generate mixing weights from the global state.
    % Absolute values enforce nonnegative weights so that dQtot/dQi >= 0.

    properties
        num_agents
        state_dim
        embed_dim
        hyper_hidden

        hyper_w1
        hyper_b1
        hyper_w2
        hyper_b2
    end

    methods
        function obj = QMIXMixer(num_agents,state_dim,embed_dim,hyper_hidden)
            obj.num_agents = num_agents;
            obj.state_dim = state_dim;
            obj.embed_dim = embed_dim;
            obj.hyper_hidden = hyper_hidden;

            obj.hyper_w1 = QMIXMLP(state_dim,hyper_hidden,embed_dim*num_agents);
            obj.hyper_b1 = QMIXMLP(state_dim,[],embed_dim);
            obj.hyper_w2 = QMIXMLP(state_dim,hyper_hidden,embed_dim);
            obj.hyper_b2 = QMIXMLP(state_dim,hyper_hidden,1);
        end

        function qtot = forward(obj,q_agents,state)
            [qtot,~] = obj.forward_cache(q_agents,state);
        end

        function [qtot,cache] = forward_cache(obj,q_agents,state)
            q_agents = double(q_agents(:));
            state = double(state(:));

            [raw_w1,c_w1] = obj.hyper_w1.forward_cache(state);
            [b1,c_b1] = obj.hyper_b1.forward_cache(state);
            [raw_w2,c_w2] = obj.hyper_w2.forward_cache(state);
            [b2,c_b2] = obj.hyper_b2.forward_cache(state);

            W1 = reshape(abs(raw_w1),obj.embed_dim,obj.num_agents);
            W2 = abs(raw_w2(:));

            z = W1*q_agents+b1;
            h = obj.elu(z);
            qtot = W2'*h+b2(1);

            cache.q_agents = q_agents;
            cache.raw_w1 = raw_w1;
            cache.raw_w2 = raw_w2;
            cache.W1 = W1;
            cache.W2 = W2;
            cache.z = z;
            cache.h = h;
            cache.c_w1 = c_w1;
            cache.c_b1 = c_b1;
            cache.c_w2 = c_w2;
            cache.c_b2 = c_b2;
        end

        function [grad_q_agents,grads] = backward(obj,cache,grad_qtot)
            g = double(grad_qtot);

            grad_W2 = cache.h*g;
            grad_h = cache.W2*g;
            grad_b2 = g;

            grad_z = grad_h.*obj.elu_derivative(cache.z);

            grad_W1 = grad_z*cache.q_agents';
            grad_b1 = grad_z;
            grad_q_agents = cache.W1'*grad_z;

            grad_raw_w1 = reshape(grad_W1,[],1).*sign(cache.raw_w1);
            grad_raw_w2 = grad_W2(:).*sign(cache.raw_w2);

            [~,g_w1] = obj.hyper_w1.backward(cache.c_w1,grad_raw_w1);
            [~,g_b1] = obj.hyper_b1.backward(cache.c_b1,grad_b1);
            [~,g_w2] = obj.hyper_w2.backward(cache.c_w2,grad_raw_w2);
            [~,g_b2] = obj.hyper_b2.backward(cache.c_b2,grad_b2);

            grads = struct();
            grads.w1 = g_w1;
            grads.b1 = g_b1;
            grads.w2 = g_w2;
            grads.b2 = g_b2;
        end

        function grads = zero_grads(obj)
            grads.w1 = obj.hyper_w1.zero_grads();
            grads.b1 = obj.hyper_b1.zero_grads();
            grads.w2 = obj.hyper_w2.zero_grads();
            grads.b2 = obj.hyper_b2.zero_grads();
        end

        function total = add_grads(~,total,addend)
            % Accumulate gradients layer by layer for compatibility with QMIXMLP.
            fields = {'w1','b1','w2','b2'};
            for ff = 1:numel(fields)
                name = fields{ff};
                for ll = 1:numel(total.(name))
                    total.(name){ll}.W = ...
                        total.(name){ll}.W + addend.(name){ll}.W;
                    total.(name){ll}.b = ...
                        total.(name){ll}.b + addend.(name){ll}.b;
                end
            end
        end

        function apply_gradients(obj,grads,learning_rate,batch_scale,grad_clip)
            if nargin<5 || isempty(grad_clip), grad_clip=10; end
            obj.hyper_w1.apply_gradients(grads.w1,learning_rate,batch_scale,grad_clip);
            obj.hyper_b1.apply_gradients(grads.b1,learning_rate,batch_scale,grad_clip);
            obj.hyper_w2.apply_gradients(grads.w2,learning_rate,batch_scale,grad_clip);
            obj.hyper_b2.apply_gradients(grads.b2,learning_rate,batch_scale,grad_clip);
        end

        function copy_weights(obj,other)
            obj.hyper_w1.copy_weights(other.hyper_w1);
            obj.hyper_b1.copy_weights(other.hyper_b1);
            obj.hyper_w2.copy_weights(other.hyper_w2);
            obj.hyper_b2.copy_weights(other.hyper_b2);
        end

        function snap = get_snapshot(obj)
            snap.w1 = obj.hyper_w1.get_snapshot();
            snap.b1 = obj.hyper_b1.get_snapshot();
            snap.w2 = obj.hyper_w2.get_snapshot();
            snap.b2 = obj.hyper_b2.get_snapshot();
        end

        function restore_snapshot(obj,snap)
            obj.hyper_w1.restore_snapshot(snap.w1);
            obj.hyper_b1.restore_snapshot(snap.b1);
            obj.hyper_w2.restore_snapshot(snap.w2);
            obj.hyper_b2.restore_snapshot(snap.b2);
        end

        function reset_optimizer(obj)
            obj.hyper_w1.reset_optimizer();
            obj.hyper_b1.reset_optimizer();
            obj.hyper_w2.reset_optimizer();
            obj.hyper_b2.reset_optimizer();
        end
    end

    methods (Static)
        function y = elu(x)
            y = x;
            idx = x<=0;
            y(idx) = exp(x(idx))-1;
        end

        function d = elu_derivative(x)
            d = ones(size(x));
            idx = x<=0;
            d(idx) = exp(x(idx));
        end
    end
end
