params_deterministic = {
    'num_packages' : 50,
    'num_balers' : 2,
    'num_materials' : 3,
    'min_baling_mass' : 5,
    'material_unit_prices' : [0.5, 0.4, 0.1],
    'num_episodes' : 10000,
    'epsilon' : 0.1,
    'epsilon_scale' : 0.999,
    'gamma' : 0.95,
    'learning_rate' : 0.001,
    'log_interval' : 100,
    'reward_threshold' : 8
}

params_probabilistic = {
    'num_packages' : 50,
    'num_balers' : 2,
    'num_materials' : 3,
    'min_baling_mass' : 5,
    'material_unit_prices' : [0.5, 0.4, 0.1],
    'num_episodes' : 40000,
    'epsilon' : 0.1,
    'epsilon_scale' : 0.9999,
    'gamma' : 0.95,
    'learning_rate' : 0.001,
    'log_interval' : 400,
    'reward_threshold' : 8
}