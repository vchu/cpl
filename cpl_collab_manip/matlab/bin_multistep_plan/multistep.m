function [action, best_plan, history] = multistep(probs, slot_states, inf_k, bin_names, nowtimesec, rate, history, debug)

if ~isfield(history, 'slots')
    history.slots = [];
    history.nowtimes = [];
end
history.slots(end+1,:) = slot_states;
history.nowtimes(end+1) = nowtimesec;

planning_params

if debug
    display_arg = 'off';
    % display_arg = 'final-detailed';
else
    display_arg = 'off';
end
% opt_options = optimset('Algorithm', 'active-set', 'FinDiffRelStep', 1, 'MaxFunEvals', opt_fun_evals);
opt_options = optimset('Algorithm', 'active-set', 'DiffMinChange', 1, 'MaxFunEvals', opt_fun_evals, ...
                       'Display', display_arg);
%opt_options = optimset('Algorithm', 'active-set', 'MaxFunEvals', opt_fun_evals);

% Generate potential sequences of bin deliveries.
bin_relevances = get_bin_relevances(t, probs, slot_states, nowtimeind, endedweight, notbranchweight);
bin_relevances(bin_relevances < min_bin_relev) = -inf;
if all(bin_relevances == -inf)
    % no bins are relevant, robot should just wait
    action = 0;
    best_plan = [];
    figure(101)
    clf
    subplot(3,1,1)
    visualize_bin_activity([], [], bin_names, ...
                           history, slot_states, numbins, rate, ...
                           nowtimesec, t, max_time, inf_k);
    subplot(3,1,2)
    visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                        nowtimesec, nowtimeind, max_time);
    subplot(3,1,3)
    visualize_cost_funs(t, probs, zeros(1,numbins), undo_dur, undo_dur_ind, nowtimesec, nowtimeind, max_time);
    return
end
deliv_seqs = gen_deliv_seqs(bin_relevances, max_beam_depth);
% These sequences are based on a beam search through
% bins not in the workspace currently and weighted using a heuristic which prefers bins
% whose expected start time is closer in the future, has not yet ended, and whose
% branch probability is high.
% t              : The full time vector [0..Tend]
% beam_counts    : The number of different bins to consider at each step of the beam search.
%                  If the first number is 3, the search will consider the top 3 bins as the
%                  first step. If the second number is 2, it will consider the top 2 bins
%                  (once the 1st choice is removed).  The length of this vector is the
%                  depth of the delivery sequence.
% probs          : Probability values for the bin step distributions.
% slot_states    : The state of the workspace slots (0 means slot is empty, >0 is the bin ID filling)
% nowtimeind     : The t-index of the current time.
% endedweight    : The penalty weight representing the number of seconds to penalize 
%                  the bin if the probability the bin has ended is 0.5
% notbranchweight: The penalty weight representing the number of seconds to penalize 
%                  the bin if the probability the bin is on this branch is 0.5

for bin_ind = 1:numbins
    binprob = sum(probs{bin_ind,1});
    startprobs = probs{bin_ind,1} / binprob;
    endprobs = probs{bin_ind,2} / binprob;
    lastrmind = nowtimeind; % TODO FIX THIS
    is_delivered(bin_ind) = any(bin_ind == deliv_seqs(1,:));
    rm_cost_fns(bin_ind,:) = remove_cost_precomp(t, endprobs, binprob, ...
                                                 undo_dur, is_delivered(bin_ind));
    lt_cost_fns(bin_ind,:) = late_cost_precomp(t, startprobs, endprobs, binprob, ...
                                               nowtimeind, lastrmind, undo_dur_ind);
end

all_best_times = [];
all_costs = [];
all_costs_split = [];
all_plans = [];
for i = 1:size(deliv_seqs,1)
    % create a template action plan given a delivery sequence
    plan = create_plan(slot_states, deliv_seqs(i,:));
    % will continue filling slots until they're all filled, then alternate
    % removing a generic bin (determined in optimization) and filling a specified bin
    % in the delivery order

    % create the durations (s) for each action step, given a plan
    % durations = create_durations(plan, traj_dur);
    durations = traj_dur*2 * ones(1,numel(plan));
    
    % optimize the opt_cost_fun for a given plan over the timings each action is completed
    % The solution is bounded below by the start time of the first action, given
    % it is executed now, and the duration of it and subsequent actions, given they execute
    % as soon as the last action is completed
    lower_bounds = [nowtimeind, durations(1:end-1)*rate];

    % the end time of the last action should be before the end of the distribution
    A = ones(1, numel(lower_bounds));
    b = numel(t)-durations(end)*rate;
    best_cost = inf;
    for start_off = 0:10:0
        x_start = lower_bounds;
        x_start(1) = x_start(1) + start_off;
        % x_start = x_start + 20*rate;
        x_sol = fmincon(@(x) opt_cost_fun(x, slot_states, plan, ...
                                          rm_cost_fns, lt_cost_fns, traj_dur_ind, 0), ...
                        x_start, ...
                        A, b, ...
                        [], [], ...
                        lower_bounds, [], ...
                        [], opt_options);
        cur_times = cumsum(x_sol / rate);

        % given the optimal timings, find the actual plan and its cost from the optimization function
        % this will fill in the bin removals from the original plan
        [cost, cur_plan, cur_costs] = opt_cost_fun(x_sol, slot_states, plan, rm_cost_fns, lt_cost_fns, traj_dur_ind, 1);
        if cost < best_cost
            best_cost = cost;
            best_costs = cur_costs;
            best_times = cur_times;
            best_plan = cur_plan;
        end
    end

    deliver_sequence = deliv_seqs(i,:);
    all_best_times(i,:) = best_times;
    all_costs(i) = best_cost;
    all_costs_split(i,:) = best_costs;
    all_plans(i,:) = best_plan;
end

actions = [];
all_plans_sorted = [];
all_action_starts = [];
all_action_ends = [];
[costs_sorted, cost_inds] = sort(all_costs);
for i = 1:size(deliv_seqs,1)
    ind = cost_inds(i);
    cost = all_costs(ind);
    best_times = all_best_times(ind,:);
    durations = traj_dur*2 * ones(1,numel(plan));
    plan = all_plans(ind,:);
    costs_split = all_costs_split(ind,:);
    action_starts = best_times;
    action_ends = best_times+durations;

    all_plans_sorted(i,:) = plan;
    all_action_starts(i,:) = action_starts;
    all_action_ends(i,:) = action_ends;

    % if action == 0, wait
    % if action  > 0, deliver bin "action"
    % if action  < 0, remove bin "action"
    actions(i) = plan_action(plan, action_starts, nowtimesec, planning_cycle);

    if debug % && i == 1
        figure(100+i)
        clf
        subplot(3,1,1)
        visualize_bin_activity(plan, [action_starts', action_ends'], bin_names, ...
                               history, slot_states, numbins, rate, ...
                               nowtimesec, t, max_time, inf_k);
        subplot(3,1,2)
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time);
        subplot(3,1,3)
        visualize_cost_funs(t, probs, is_delivered, undo_dur, undo_dur_ind, nowtimesec, nowtimeind, max_time);
        if actions(i) == 0
            action_name = 'WAIT';
        elseif actions(i) > 0
            action_name = sprintf('DELIVER %s', bin_names{actions(i)});
        else
            action_name = sprintf('REMOVE %s', bin_names{-actions(i)});
        end
        title(sprintf('Cost: %.1f | Action: %s', cost, action_name))
    end
end

plan_costs = nan*zeros(size(all_plans_sorted,1)*2,size(all_plans_sorted,2)+1);
for i = 1:size(all_plans_sorted,1)
    plan_costs(2*i-1,1) = costs_sorted(i);
    plan_costs(2*i-1,2:end) = all_plans_sorted(i,:);
    plan_costs(2*i,2:end) = all_costs_split(i,:);
end
plan_costs

action = actions(1);
best_plan = [all_plans_sorted(1,:)', all_action_starts(1,:)', all_action_ends(1,:)'];