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

function [trigger_time] = judgment_trigger(time_slot,Tp,N_slot,T_d,T_c,tkhat,t0,L)
%JUDGMENT_TRIGGER Correct a candidate trigger instant according to TDM availability.
%
% If the candidate self-triggering instant lies in an active slot, it is kept
% unchanged. If it lies in an inactive slot, the trigger is moved to the last
% admissible controller-execution instant of the nearest preceding active slot.
%
% Inputs:
%   time_slot - Length of one TDM slot.
%   Tp        - Length of one TDM frame/prediction period.
%   N_slot    - Number of slots in one frame.
%   T_d       - Duration of one controller execution cycle.
%   T_c       - Reserved inactive/communication duration within a slot.
%   tkhat     - Candidate self-triggering instant.
%   t0        - Current trigger instant; the corrected instant cannot precede it.
%   L         - Binary TDM allocation sequence (1: active, 0: inactive).
%
% Output:
%   trigger_time - Admissible trigger instant after TDM correction.

scale = 1000;
slot_tick = round(time_slot*scale);
frame_tick = round(Tp*scale);
Td_tick = round(T_d*scale);
Tc_tick = round(T_c*scale);
current_tick = round(tkhat*scale);

frame_index = floor(current_tick/frame_tick);
frame_offset_tick = mod(current_tick,frame_tick);
ff = floor(frame_offset_tick/slot_tick)+1;
ff = min(max(ff,1),N_slot);
L_index = frame_index*N_slot+ff;

num_allocated_slots = size(L,2);
if L_index > num_allocated_slots
    trigger_time = current_tick/scale;
    return;
end

if L(L_index)==1
    trigger_time = current_tick/scale;
    return;
end

previous_active_slot = findOneBeforeZero(L,L_index);
if isempty(previous_active_slot)
    error(['Early triggering failed: no active slot exists before the current ' ...
           'inactive slot. Check the TDM allocation sequence.']);
end

active_tick = slot_tick-Tc_tick;
num_cycles = floor(active_tick/Td_tick);
last_offset_tick = (num_cycles-1)*Td_tick;
trigger_tick = (previous_active_slot-1)*slot_tick+last_offset_tick;
trigger_time = trigger_tick/scale;

t0_tick = round(t0*scale);
if trigger_tick < t0_tick
    error(['The corrected trigger time %.3f s precedes the current trigger ' ...
           'instant %.3f s. Check the allocation sequence and the minimum ' ...
           'triggering interval.'],trigger_time,t0_tick/scale);
end
end

function onePosition = findOneBeforeZero(arr,zeroPosition)
%FINDONEBEFOREZERO Return the nearest preceding active slot.
if zeroPosition<=1 || zeroPosition>length(arr)
    onePosition = [];
    return;
end

if arr(zeroPosition)~=0
    onePosition = [];
    return;
end

onePositions = find(arr(1:zeroPosition-1)==1);
if isempty(onePositions)
    onePosition = [];
else
    onePosition = onePositions(end);
end
end
