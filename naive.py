import numpy as np
import logging
import parameters
import environment

if __name__ == '__main__' :
    env = environment.Gateway()
    num_packages = 50

    packages = np.genfromtxt('orders_dtrm.csv', delimiter=',')
    # packages = np.genfromtxt('orders_prob.csv', delimiter=',')

    total_reward = 0.0
    pkg_id = 0

    while pkg_id < num_packages :
        if(~env.source_state.any()) :
                env.source_state = packages[pkg_id, :]
                pkg_id += 1
        
        mac_list = range(parameters.params['num_balers'])
        mat_in_mac = env.baler_state + env.source_state
        grade = np.argmax(mat_in_mac, axis=1).astype(int)

        material_unit_prices = np.asarray(parameters.params['material_unit_prices'])
        price_vec = mat_in_mac[mac_list, grade] * material_unit_prices[grade]

        empty_mac = np.where(~env.baler_state.any(axis=1))[0]

        # prioritize empty baler
        if empty_mac.size != 0 :
            price_vec[empty_mac[0]] += 1

        action = np.argmax(price_vec)
        reward = env.step(action)
        total_reward += reward

    print(total_reward)