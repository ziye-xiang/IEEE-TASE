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

classdef QMIXMLP < handle
    %QMIXMLP Lightweight fully connected network: ReLU hidden layers, linear output, and Adam.

    properties
        input_dim
        hidden_dims
        output_dim
        layers

        adam_mW
        adam_vW
        adam_mb
        adam_vb
        adam_t = 0
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-8
    end

    methods
        function obj = QMIXMLP(input_dim,hidden_dims,output_dim)
            obj.input_dim = input_dim;
            obj.hidden_dims = hidden_dims(:)';
            obj.output_dim = output_dim;

            dims = [input_dim,obj.hidden_dims,output_dim];
            L = numel(dims)-1;

            obj.layers = cell(L,1);
            obj.adam_mW = cell(L,1);
            obj.adam_vW = cell(L,1);
            obj.adam_mb = cell(L,1);
            obj.adam_vb = cell(L,1);

            for l = 1:L
                fan_in = dims(l);
                fan_out = dims(l+1);

                if l<L
                    W = randn(fan_out,fan_in)*sqrt(2/fan_in);
                else
                    W = randn(fan_out,fan_in)*sqrt(1/fan_in);
                end
                b = zeros(fan_out,1);

                obj.layers{l} = struct('W',W,'b',b);
                obj.adam_mW{l} = zeros(size(W));
                obj.adam_vW{l} = zeros(size(W));
                obj.adam_mb{l} = zeros(size(b));
                obj.adam_vb{l} = zeros(size(b));
            end
        end

        function y = forward(obj,x)
            x = double(x(:));
            a = x;
            L = numel(obj.layers);

            for l = 1:L
                z = obj.layers{l}.W*a+obj.layers{l}.b;
                if l<L
                    a = max(z,0);
                else
                    a = z;
                end
            end
            y = a;
        end

        function [y,cache] = forward_cache(obj,x)
            x = double(x(:));
            L = numel(obj.layers);

            cache.a = cell(L+1,1);
            cache.z = cell(L,1);
            cache.a{1} = x;

            a = x;
            for l = 1:L
                z = obj.layers{l}.W*a+obj.layers{l}.b;
                cache.z{l} = z;

                if l<L
                    a = max(z,0);
                else
                    a = z;
                end
                cache.a{l+1} = a;
            end
            y = a;
        end

        function [grad_input,grads] = backward(obj,cache,grad_output)
            L = numel(obj.layers);
            grads = obj.zero_grads();
            delta = double(grad_output(:));

            for l = L:-1:1
                a_prev = cache.a{l};
                grads{l}.W = delta*a_prev';
                grads{l}.b = delta;

                grad_prev = obj.layers{l}.W'*delta;
                if l>1
                    delta = grad_prev.*(cache.z{l-1}>0);
                else
                    delta = grad_prev;
                end
            end
            grad_input = delta;
        end

        function grads = zero_grads(obj)
            L = numel(obj.layers);
            grads = cell(L,1);
            for l = 1:L
                grads{l} = struct( ...
                    'W',zeros(size(obj.layers{l}.W)), ...
                    'b',zeros(size(obj.layers{l}.b)));
            end
        end

        function apply_gradients(obj,grads,learning_rate,batch_scale,grad_clip)
            if nargin<4 || isempty(batch_scale), batch_scale=1; end
            if nargin<5 || isempty(grad_clip), grad_clip=10; end

            sqnorm = 0;
            for l = 1:numel(grads)
                gW = grads{l}.W*batch_scale;
                gb = grads{l}.b*batch_scale;
                sqnorm = sqnorm+sum(gW(:).^2)+sum(gb(:).^2);
            end
            gnorm = sqrt(sqnorm);

            clip_scale = 1;
            if isfinite(gnorm) && gnorm>grad_clip
                clip_scale = grad_clip/(gnorm+1e-12);
            end

            obj.adam_t = obj.adam_t+1;
            t = obj.adam_t;

            for l = 1:numel(grads)
                gW = grads{l}.W*batch_scale*clip_scale;
                gb = grads{l}.b*batch_scale*clip_scale;

                obj.adam_mW{l} = obj.beta1*obj.adam_mW{l}+(1-obj.beta1)*gW;
                obj.adam_vW{l} = obj.beta2*obj.adam_vW{l}+(1-obj.beta2)*(gW.^2);
                obj.adam_mb{l} = obj.beta1*obj.adam_mb{l}+(1-obj.beta1)*gb;
                obj.adam_vb{l} = obj.beta2*obj.adam_vb{l}+(1-obj.beta2)*(gb.^2);

                mW_hat = obj.adam_mW{l}/(1-obj.beta1^t);
                vW_hat = obj.adam_vW{l}/(1-obj.beta2^t);
                mb_hat = obj.adam_mb{l}/(1-obj.beta1^t);
                vb_hat = obj.adam_vb{l}/(1-obj.beta2^t);

                obj.layers{l}.W = obj.layers{l}.W-learning_rate* ...
                    mW_hat./(sqrt(vW_hat)+obj.adam_eps);
                obj.layers{l}.b = obj.layers{l}.b-learning_rate* ...
                    mb_hat./(sqrt(vb_hat)+obj.adam_eps);
            end
        end

        function copy_weights(obj,other)
            for l = 1:numel(obj.layers)
                obj.layers{l}.W = other.layers{l}.W;
                obj.layers{l}.b = other.layers{l}.b;
            end
        end

        function snap = get_snapshot(obj)
            snap = cell(numel(obj.layers),1);
            for l = 1:numel(obj.layers)
                snap{l} = struct('W',obj.layers{l}.W,'b',obj.layers{l}.b);
            end
        end

        function restore_snapshot(obj,snap)
            for l = 1:numel(obj.layers)
                obj.layers{l}.W = snap{l}.W;
                obj.layers{l}.b = snap{l}.b;
            end
        end

        function reset_optimizer(obj)
            obj.adam_t = 0;
            for l = 1:numel(obj.layers)
                obj.adam_mW{l}(:) = 0;
                obj.adam_vW{l}(:) = 0;
                obj.adam_mb{l}(:) = 0;
                obj.adam_vb{l}(:) = 0;
            end
        end
    end
end
