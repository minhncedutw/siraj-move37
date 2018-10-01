import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(utility_grids, policy_grids):

    return 0

if __name__ == '__main__':
    shape = (3, 4)
    goal = (0, -1)
    trap = (1, -1)
    obstacle = (1, 1)
    start = (2, 0)
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1

    reward_grid = np.zeros(shape=shape) + default_reward
    reward_grid[goal] = goal_reward
    reward_grid[trap] = trap_reward
    reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal] = True
    terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask[1, 1] = True

    gw = GridWorldMDP(reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                          (-1, 0.1),
                          (0, 0.8),
                          (1, 0.1),
                      ],
                      no_action_probability=0.0)

    mdp_solvers = {
        'Value Iteration': gw.run_value_iterations,
        'Policy Iteration': gw.run_policy_iterations
    }

    for solver_name, solver_fn in mdp_solvers.items():
        print('Final result of {}:'.format(solver_name))
        policy_grids, utility_grids = solver_fn(iterations=25, discount=0.5)
        print(policy_grids[:, :, -1])
        print(utility_grids[:, :, -1])
        plt.figure()
        gw.plot_policy(utility_grids[:, :, -1])
        plot_convergence(utility_grids=utility_grids, policy_grids=policy_grids)
        plt.show()

    ql = QLearner(num_states=)