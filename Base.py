from get_data import split_data


class Base:
    def __init__(self):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data()


    def generate_params(self):
        pass


    def fit(self):
        pass


        
