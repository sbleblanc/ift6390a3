    def earlyStopping(self, training_set_proportion, validation_set_proportion, testing_set_proportion):
        first_separator = len(self.data_points)*training_set_proportion
        second_separator = len(self.data_points) - len(self.data_points)*testing_set_proportion
        self.training_set = self.data_points[:first_separator]
        self.validation_set = self.data_points[first_separator:second_seperator]
        self.testing_set = self.data_points[second_separator:]
        
        epoch = 0
        last_cost = 0.
        current_cost = 1.
        while (last_cost - current_cost) > 0:
            last_cost = current_cost
            epoch += 1
            for i in range(0, len(self.training_set), self.hyper_params.K):
                end_batch = min(i + self.hyper_params.K, len(self.training_set))
                batch_data = self.training_set[i:end_batch]
                bl = self.single_batch(batch_data)
                self.params.W1 -= self.hyper_params.eta * bl.W1
                self.params.W2 -= self.hyper_params.eta * bl.W2
                self.params.b1 -= self.hyper_params.eta * bl.b1
                self.params.b2 -= self.hyper_params.eta * bl.b2
            current_cost = self.current_cost_validating()
            print('Epoch {}: Loss={}'.format(epoch, current_cost))
            
    def current_cost_validating(self):
        cummul = 0.
        for dp in self.validating_set:
            fl = self.fprop(dp.x)
            cummul += -np.log(fl.os[int(dp.y)])
        cummul /= len(self.validating_set)
        return cummul
