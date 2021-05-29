from time import time
import random


### Global Settings

# Print formatting
name_alias_max = 6
grid_padding = 10
print_name_alias = True

# MDP config
grid_rows, grid_cols = 1, 4
# convergence_tol = 0.01
discount_factor = 0.9


### MDP classes

class State:
    def __init__(self, pos, is_absorb_state, reward, name_alias=None):
        assert len(name_alias) <= name_alias_max if name_alias is not None else True  # Allow single char name only
        self.pos = pos
        self.is_absorb_state = is_absorb_state
        self.reward = reward
        self.name_alias = name_alias
        self.value = 0
        self.value_old = 0   # Not used
        self.is_converged = True
        self.policy = "--"

class Transition:
    def __init__(self, from_location, to_location, reward, transition_prob):
        self.from_location = from_location
        self.to_location = to_location
        self.reward = reward
        self.transition_prob = transition_prob

class Action:
    def __init__(self, action, transitions):
        self.action = action
        self.transitions = transitions


### Print functions

def print_states(states, value_to_print, print_name_alias):
    if value_to_print == "value":
        f_print_value = lambda state: state.value
    elif value_to_print == "policy":
        f_print_value = lambda state: state.policy
    else:
        raise ValueError("value_to_print must be one of {0}".format(["value", "policy"]))

    for i in reversed(range(1, grid_rows+1)):   # Print like a "mathematical graph"
        out_str = ""
        for j in range(1, grid_cols+1):
            pos = (i, j)
            if pos in states:
                state = states[pos]
                s = str(f_print_value(state))[:name_alias_max]
                if print_name_alias and state.is_absorb_state:
                    s = str(state.name_alias)
            else:
                s = ""
            out_str += s.ljust(grid_padding)
        print(out_str)

def print_states_value(states):
    print_states(states, "value", print_name_alias)

def print_states_policy(states):
    print_states(states, "policy", print_name_alias)

def get_actions(actions, actions_rule, pos):
    avai_actions = actions_rule[pos]
    filtered_actions = { key: actions[key] for key in avai_actions }
    return filtered_actions


### Main

def main():
    t0 = time()

    ### Create states
    states = {}
    for i in range(1, grid_rows+1):
        for j in range(1, grid_cols+1):
            pos = (i,j)
            states[pos] = State(pos, is_absorb_state=False, reward=0)

    ### Create actions
    
    actions = {}
    actions["PICKUP"] = Action("PICKUP", transitions=[
        Transition((1,1), (1,1), 0, 0.4), Transition((1,1), (1,2), 9, 0.12), Transition((1,1), (1,3), 13.5, 0.18), Transition((1,1), (1,4), 10.75, 0.3),
        Transition((1,2), (1,1), 10, 0.24), Transition((1,2), (1,2), 0, 0.7), Transition((1,2), (1,3), 7.25, 0.06),
        Transition((1,3), (1,1), 11.5, 0.2), Transition((1,3), (1,3), 0, 0.6), Transition((1,3), (1,4), 8.2, 0.2),
        Transition((1,4), (1,1), 9.75, 0.42), Transition((1,4), (1,2), 4, 0.28), Transition((1,4), (1,4), 0, 0.3),
    ])
    actions["TOL1"] = Action("TOL1", transitions=[
        Transition((1,2), (1,1), -1, 1), Transition((1,3), (1,1), -1.5, 1), Transition((1,4), (1,1), -1.25, 1)
    ])
    actions["TOL2"] = Action("TOL2", transitions=[
        Transition((1,1), (1,2), -1, 1), Transition((1,4), (1,2), -1, 1)
    ])
    actions["TOL3"] = Action("TOL3", transitions=[
        Transition((1,1), (1,3), -1.5, 1), Transition((1,2), (1,3), -0.75, 1)
    ])
    actions["TOL4"] = Action("TOL4", transitions=[
        Transition((1,1), (1,4), -1.25, 1), Transition((1,3), (1,4), -0.8, 1)
    ])


    ### Policy Iteration
    # Random initial policies
    actions_rule = {
        (1,1): ['PICKUP', 'TOL2', 'TOL3', 'TOL4'],
        (1,2): ['PICKUP', 'TOL1', 'TOL3'],
        (1,3): ['PICKUP', 'TOL1', 'TOL4'],
        (1,4): ['PICKUP', 'TOL1', 'TOL2']
    }

    for state in states.values():
        if state.is_absorb_state: continue
        state.policy = random.choice(actions_rule[state.pos])
    print_states_value(states)
    print()
    print_states_policy(states)
    print("Initial random state policies" + "\n")

    i = 0
    while True:
        # Estimate value for current policy
        for state in states.values():
            if state.is_absorb_state: continue
            expected_util = 0
            action = actions[state.policy]
            avail_transitions = [t for t in action.transitions if t.from_location==state.pos]
            for t in avail_transitions:
                pos_new = t.to_location
                state_new = states[pos_new]
                expected_util += t.transition_prob * (t.reward + (state_new.value * discount_factor))
            state.value = expected_util
        # Policy iteration
        for state in states.values():
            if state.is_absorb_state: continue
            expected_utils = []
            for action in get_actions(actions, actions_rule, state.pos).values():
                expected_util = 0
                avail_transitions = [t for t in action.transitions if t.from_location==state.pos]
                for t in avail_transitions:
                    pos_new = t.to_location
                    state_new = states[pos_new]
                    expected_util += t.transition_prob * (t.reward + (state_new.value * discount_factor))
                expected_utils.append((expected_util, action.action))
            best_action = max(expected_utils, key=lambda x: x[0])  # Action with max value / expected utility
            policy_new = best_action[1]
            state.is_converged = state.policy == policy_new
            if not state.is_converged:
                state.policy = policy_new
        # Print
        i += 1
        print_states_value(states)
        print()
        print_states_policy(states)
        print("Iteration: {0}".format(i) + "\n")
        # Convergence check
        is_converged = all([state.is_converged for state in states.values()])
        if is_converged: break

    print_states_policy(states)
    print("Best policy" + "\n")
    print("Time taken is {:.4f} seconds".format(time() - t0))


if __name__ == "__main__":
    main()