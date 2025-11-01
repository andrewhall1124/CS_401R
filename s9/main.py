from gridworld import GridWorld
from players import (
    SARSAAgent,
    QLearningAgent,
    TDLambdaAgent,
    OptimalAgent,
    plot_rewards,
    print_policy_q,
    print_policy_v,
    print_policy_optimal
)


env = GridWorld()
num_episodes = 1000

# # SARSA Agent
# sarsa_agent = SARSAAgent(env)
# sarsa_q_values, sarsa_rewards = sarsa_agent.train(num_episodes)

# # Q-Learning Agent
# q_learning_agent = QLearningAgent(env)
# q_learning_q_values, q_learning_rewards = q_learning_agent.train(num_episodes)

# TD(λ) Agent
# eta_func = lambda x, t, N_visits: 0.9 / (1 + N_visits[x])
td_lambda_agent = TDLambdaAgent(env, lam=0.7, eta=0.3)
td_lambda_v_values, td_lambda_rewards = td_lambda_agent.train(num_episodes)

# Optimal Agent (planning)
optimal_agent = OptimalAgent(env)
V_opt, pi_opt = optimal_agent.value_iteration()
optimal_rewards = optimal_agent.evaluate(num_episodes)

# # Plotting the rewards
# plot_rewards(
# #     # [sarsa_rewards, q_learning_rewards, td_lambda_rewards, optimal_rewards],
# #     # ["SARSA", "Q-Learning", f"TD(λ), λ={td_lambda_agent.lam}", "Optimal"],
#     [td_lambda_rewards],
#     [f"λ={td_lambda_agent.lam}"],
# )

# Print Policies
# print("Policy derived from SARSA:")
# print_policy_q(sarsa_q_values, env)
# print("\nPolicy derived from Q-Learning:")
# print_policy_q(q_learning_q_values, env)
print("\nPolicy derived from TD(λ):")
print_policy_v(td_lambda_v_values, env)

print("\nOptimal policy (from value iteration):")
print_policy_optimal(pi_opt, env)
