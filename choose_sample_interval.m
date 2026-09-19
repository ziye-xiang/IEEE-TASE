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

function [T,cc,ee,ff,gg] = choose_sample_interval(time_slot,Tp,N,T_d,T_c,tkhat)
%CHOOSE_SAMPLE_INTERVAL Compute the next admissible execution interval in a TDM frame.
%
% Inputs:
%   time_slot - Length of one TDM slot.
%   Tp        - Length of one TDM frame/prediction period.
%   N         - Number of slots in one frame.
%   T_d       - Duration of one controller execution cycle.
%   T_c       - Reserved inactive/communication duration within a slot.
%   tkhat     - Current candidate time instant.
%
% Outputs:
%   T  - Time increment to the next admissible execution instant.
%   cc - Execution-cycle index inside the current active part of the slot.
%   ee - Number of admissible execution instants in one frame.
%   ff - Current slot index inside the frame.
%   gg - Global execution-instant index.
%
% Integer ticks are used to avoid floating-point ambiguity in slot indexing.

scale = 1000;
slot_tick = round(time_slot*scale);
frame_tick = round(Tp*scale);
Td_tick = round(T_d*scale);
Tc_tick = round(T_c*scale);
current_tick = round(tkhat*scale);

active_tick = slot_tick-Tc_tick;
bb = floor(active_tick/Td_tick);
if bb < 1
    error('Invalid TDM parameters: the active part of a slot must contain at least one T_d cycle.');
end

offset_tick = mod(current_tick,slot_tick);
valid_offsets = (0:bb-1)*Td_tick;
idx = find(offset_tick==valid_offsets,1,'first');
if isempty(idx)
    error(['Invalid TDM instant %.6f s (slot offset %.3f s). ' ...
           'Admissible offsets are %s s.'], ...
          tkhat,offset_tick/scale,mat2str(valid_offsets/scale));
end
cc = idx;

if cc < bb
    next_tick = current_tick+Td_tick;
else
    slot_start_tick = current_tick-offset_tick;
    next_tick = slot_start_tick+slot_tick;
end
T = (next_tick-current_tick)/scale;

frame_offset_tick = mod(current_tick,frame_tick);
ff = floor(frame_offset_tick/slot_tick)+1;
ff = min(max(ff,1),N);

ee = bb*N;
frame_index = floor(current_tick/frame_tick);
gg = (ff-1)*bb+cc+ee*frame_index;
end
