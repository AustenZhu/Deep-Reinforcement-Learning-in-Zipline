from zipdl.models import DDQN_agent
from zipdl.envs import dynamic_beta_env
from zipdl.algos import multifactor as mf

BATCH_SIZE = 32
EPISODES = 5000

algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
env = dynamic_beta_env()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DDQN_agent(state_size, action_size)
done = False

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
