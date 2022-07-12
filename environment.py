import numpy as np

class Gateway :
    def __init__(self, params=None) :
        # check if params is in correct format
        assert type(params) == dict

        # enviornment parameters
        self.params = params

        # vector denoting mass of each material in source
        self.source_state_space = (self.params['num_materials'])
        # vector denoting mass of each material in each machine
        self.baler_state_space = (self.params['num_balers'], self.params['num_materials'])
        
        # state space is [source_state, baler_state.flatten()]
        self.state_space = (self.source_state_space + self.baler_state_space[0] * self.baler_state_space[1])
        # action for sending source package to baler
        self.action_space = self.params['num_balers']
        
        # define state valiables
        self.resetAll()
    
    def resetSource(self) :
        self.source_state *= 0.0

    def resetBaler(self, baler) :
        self.baler_state[baler, :] *= 0.0

    def resetAll(self) :
        self.source_state = np.zeros(self.source_state_space) 
        self.baler_state = np.zeros(self.baler_state_space)

    def step(self, baler) :
        # add source package to baler
        self.baler_state[baler, :] += self.source_state
        self.resetSource()
        reward = 0.0

        # check if any baler is filled
        baler_mass = np.sum(self.baler_state, axis=1)
        filled_baler = np.where(baler_mass > self.params['min_baling_mass'])[0]

        # make bale if baler is filled
        if filled_baler.size != 0 :
            bale_grade = np.argmax(self.baler_state[filled_baler[0]])
            reward = self.params['material_unit_prices'][bale_grade] * self.baler_state[filled_baler[0], bale_grade]
            self.resetBaler(filled_baler[0])

        return reward
    
    def state(self) :
        return np.append(self.source_state, self.baler_state.flatten())