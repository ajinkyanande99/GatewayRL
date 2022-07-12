import parameters
import environment
import algorithms

if __name__ == '__main__' :
    # define environment
    params = parameters.params_probabilistic
    env = environment.Gateway(params=params)

    # q_learning
    algorithms.q_learning(environment=env, composition='Probabilistic',
                          package_order_shuffling=False, exploration='Exponential',
                          save_weights=True, save_returns=True)