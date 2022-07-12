import numpy as np

def action(environment, q, epsilon) :
    act = np.argmax(q)
    prob_of_act = np.random.rand()
    
    if prob_of_act < epsilon :
        act = np.random.randint(environment.action_space)
    
    return act

def packageOrderShuffling(environment, packages) :
    seq = np.arange(environment.params['num_packages'])
    np.random.shuffle(seq)
    packages = packages[seq, :]

    return packages

def decayEpsilonExp(environment, epsilon) :
    return epsilon * environment.params['epsilon_scale']

def q_learning(environment, composition, package_order_shuffling, exploration, save_weights=False, save_returns=False) :
    # initialize
    weights = np.zeros((environment.state_space, environment.action_space))
    b = 0.0
    epsilon = 0.0
    avg_total_reward = 0.0

    # constant exploration
    if exploration == 'Constant' :
        epsilon = environment.params['epsilon']

    # decaying exploration : initialize exploration to 1 and decay with epsilon scale to near 0
    if exploration == 'Exponential' :
        epsilon = 1.0

    # log returns
    if save_returns :
        returns = []

    # iterate over episodes
    for episode in range(environment.params['num_episodes']) :
        # reset env before entering first package of episode
        environment.resetAll()

        # import package data
        if composition == 'Deterministic' :
            packages = np.genfromtxt('orders_deterministic.csv', delimiter=',')
            assert packages.shape[0] == environment.params['num_packages']
        
        if composition == 'Probabilistic' :
            packages = np.genfromtxt('orders_probabilistic.csv', delimiter=',')
            assert packages.shape[0] == environment.params['num_packages']

        # package order shuffling
        if package_order_shuffling :
            packages = packageOrderShuffling(environment, packages)

        # initialize
        total_reward = 0.0
        pkg_id = 0

        while pkg_id < environment.params['num_packages'] :
            # iterate over incoming packages individually without future information
            if(~environment.source_state.any()) :
                environment.source_state = packages[pkg_id, :]
                pkg_id += 1
            
            state = environment.state()
            q = (state.T @ weights + b).T

            act = action(environment, q, epsilon)
            reward = environment.step(act)
            total_reward += reward

            state_prime = environment.state()
            q_prime = (state_prime.T @ weights + b).T

            grad_b  = 1.0
            grad_weights = state

            fac = environment.params['learning_rate'] * (q[act] - (reward + environment.params['gamma'] * np.max(q_prime)))
            
            weights[:, act] -= fac * grad_weights.flatten()
            b -= fac * grad_b

        avg_total_reward += total_reward

        # exponential epsilon decay
        if exploration == 'Exponential' :
            epsilon = decayEpsilonExp(environment, epsilon)

        # logging
        if (episode + 1) % environment.params['log_interval'] == 0 :
            avg_total_reward /= environment.params['log_interval']
            print('episode : ', episode + 1)
            print('returns : ', avg_total_reward)
            print('epsilon : ', epsilon)
            returns.append(avg_total_reward)
            avg_total_reward = 0.0

    if save_weights :
        weights = weights.flatten()
        weights = np.append(np.array([b]), weights)
        np.savetxt('weights.txt', weights, delimiter='\n')

    if save_returns :
        np.savetxt('returns.txt', returns, delimiter='\n')