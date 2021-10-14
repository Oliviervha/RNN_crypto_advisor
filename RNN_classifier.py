import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from config import Configuration 
import numpy as np
from sklearn.metrics import classification_report


config = Configuration() # configuration object

class Data:

    def __init__(self):

        self.df = pd.DataFrame()

    def read(self):
        '''
        ### Read datafile

        Notse: 
        - dataset must be ascending by date
        - missing data will lead to no output

        '''
        try:
            #read data from csv 
            self.df = pd.read_csv(config.PATHTOFILE, sep = ';') 

        except:
            #read data from xlsx
            self.df = pd.read_excel(config.PATHTOFILE) 

class RNN:

    def __init__(self, df):

        self.df = df.copy()
        self.df_norm = df.copy()

        self.X_train = []
        self.X_valid = []
        self.X_test = []

        self.y_train = []
        self.y_valid = []
        self.y_test = []

        self.model = None

        self.test_pred = []
        self.test_pred_flat = []
        self.y_test_flat = []

        self.total_buy_points = 0


    def normalizeInput(self, scaler = False):
        
        for col in config.FEATURES:
            p = [0] # set base to 0
            # normalize data by taking the relative difference t from t-1
            for i in range(1,len(self.df_norm)):
                p.append((self.df_norm[col].iloc[i] / self.df_norm[col].iloc[i-1]) - 1) # (p_i / p_(i-1)) - 1

            self.df_norm[col] = p


    def TrainTestSplit(self):

        '''
        Time series split

        Provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. 
        In each split, test indices must be higher than before, and thus shuffling in cross validator is inappropriate.

        '''
        X = []
        y = []


        tscv = TimeSeriesSplit(gap=0, max_train_size = config.INPUT_SIZE, n_splits=int(len(self.df_norm)/config.OUTPUT_SIZE) - int(config.INPUT_SIZE / config.OUTPUT_SIZE), test_size = config.OUTPUT_SIZE)
        
        for X_index, y_index in tscv.split(self.df_norm[config.PRICE_COL]):

            #X.append([self.df_norm[config.PRICE_COL].iloc[i] for i in X_index])
            X.append([[self.df_norm[col].iloc[i] for col in config.FEATURES] for i in X_index]) #, [self.df_norm[config.VOL_COL].iloc[i] for i in X_index]])
            y.append([[[self.df_norm[config.PRICE_COL].iloc[i]]] for i in y_index])
            

        for i in range(len(y)):
            
            # target variable becomes 1 if price increases by min_increase otherwise (classification)
            if any(p > config.MIN_PRICE_INCREASE for p in y[i][0][0]):
                y[i][0][0] = 1
                
            else:
                y[i][0][0] = 0


        '''
        Train / Validation / Test set

            With a validation set, you're essentially taking a fraction of your samples out of your training set, 
            or creating an entirely new set all together, and holding out the samples in this set from training.
            During each epoch, the model will be trained on samples in the training set but will NOT be trained on samples in the validation set. 
            Instead, the model will only be validating on each sample in the validation set.

            The purpose of doing this is for you to be able to judge how well your model can generalize. 
            Meaning, how well is your model able to predict on data that it's not seen while being trained.

        '''

        total_sequences = len(X)

        # define train, val, test sizes from config
        train_size = int(total_sequences * config.TRAIN_SIZE)
        val_size = train_size + int(total_sequences * config.VAL_SIZE)
        test_size = val_size + int(total_sequences * config.TEST_SIZE)

        # define train, val, test sets for X and y
        self.X_train, self.y_train = np.array(X[:train_size]), np.array(y[:train_size])
        self.X_valid, self.y_valid = np.array(X[train_size:val_size]), np.array(y[train_size:val_size])
        self.X_test, self.y_test = np.array(X[val_size:test_size]), np.array(y[val_size:test_size])

        # print("Training: {}, Valid: {}, Test: {}".format(len(self.X_train), len(self.X_valid), len(self.X_test)))

        # reshape to 3 dimensional -> (batch x timesteps x features) required for RNN input
        self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], self.y_train.shape[1], 1)).astype(np.float32)
        self.y_valid = np.reshape(self.y_valid, (self.y_valid.shape[0], self.y_valid.shape[1], 1)).astype(np.float32)
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], self.y_test.shape[1], 1)).astype(np.float32)

        if self.X_train.ndim < 3: # if only 1 feature as input
            self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1)).astype(np.float32)
            self.X_valid = np.reshape(self.X_valid, (self.X_valid.shape[0], self.X_valid.shape[1], 1)).astype(np.float32)
            self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1)).astype(np.float32)

        # print(self.X_train.shape, self.y_train.shape)

    def trainModel(self):

        '''
            We'll define a sequential model and add the SimpleRNN layer by defining the input shapes. 
            We'll add Dense layers with ReLU activations, set the output layer dimension, and compile the model with Adam optimizer.

            Source: https://www.datatechnotes.com/2020/01/multi-output-multi-step-regression.html

        '''

        self.model = tf.keras.models.Sequential() # initialize sequential model

        self.model.add(tf.keras.layers.SimpleRNN(64, input_shape = (config.INPUT_SIZE, config.N_FEATURES), activation = 'sigmoid', return_sequences = False)) # add input layer
        self.model.add(tf.keras.layers.Dense(64, activation="sigmoid")) # add dense layer with X internal units (neurons)
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid")) # add dense layer as output layer

        opt = tf.keras.optimizers.Adam(learning_rate = config.LR) # define optimizer
        self.model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

        # self.model.summary() # print model summary
        

        self.model.fit(self.X_train, self.y_train, epochs = config.EPOCHS, validation_data = (self.X_valid, self.y_valid), verbose = 0) # train model

        # self.model.evaluate(self.X_valid, self.y_valid) # evaluate model

        self.test_pred = self.model.predict(self.X_test) # predict test set

        if config.SAVEMODEL:
            self.model.save('./models')




    def evaluate(self):

        '''
        # Classification report

        Metrics to evaluate a classification problem. 
        Especially the precision of the buy class is of interest, as if we buy we want to make sure that we can sell with profit the other day. 
        The fraction provides the fraction of this over the test set. 

        '''
        self.y_test = np.array(self.y_test)[:,0,:] # remove 3th dimension from test set

        y_pred = []
        for l in self.test_pred:

            if l[0] > config.MIN_PROBA:
                y_pred.append([1])
                self.total_buy_points += 1

            else:
                y_pred.append([0])
    
        # print(classification_report(self.y_test, y_pred, target_names = ['hold', 'buy']))

        '''
        Visualize predictions test set

        '''
        test_act = self.df[config.PRICE_COL].tail(len(self.test_pred)).tolist()

        test_pred = []
        test_pred_idx = []
        inv_return = 0
        reinv_return = 1
        for i in range(len(self.test_pred)):

            if self.test_pred[i] > config.MIN_PROBA:
                test_pred_idx.append(i)
                test_pred.append(test_act[i])

                if i < len(self.test_pred)-1:
                    inv_return += ((test_act[i+1] - test_act[i]) / test_act[i])
                    reinv_return = reinv_return * (1+((test_act[i+1] - test_act[i]) / test_act[i]))

            self.test_pred_flat.extend(self.test_pred[i])
            self.y_test_flat.extend(self.y_test[i])

        # print trend
        trend = (test_act[-1] - test_act[0]) / test_act[0] * 100
        print('Trend: {}%'.format(round(trend,2)))

        plt.figure(0)
        plt.plot([i for i in range(len(test_act))], test_act, label = 'Actual')
        plt.scatter([idx for idx in test_pred_idx], test_pred, label = 'Buy point', color = 'green')

        # Set the x & y axis labels
        plt.xlabel('t')
        plt.ylabel('Price ($)')

        # Set a title of the current axes.
        print('RNN investment return: {}%'.format(round(inv_return*100,2)))
        print('RNN reinvestment return: {}%'.format(round((reinv_return-1)*100,2)))
        plt.title('Buy point recognition using RNN\nStrategy: selling next day\nTrend: {}%\nProb. > {}%, Return: {}%, Reinv. return: {}%'.format(round(trend,2), round(config.MIN_PROBA*100,0), round(inv_return*100,2), round((reinv_return-1)*100,2)))

        # show a legend on the plot
        plt.legend()
        plt.grid()
        plt.savefig('plot.png')
        # plt.show()


    def baseline(self):
        '''
        Visualize baseline

        '''
        test_act = self.df[config.PRICE_COL].tail(len(self.test_pred)).tolist()
        plt.figure(1)
        plt.plot([i for i in range(len(test_act))], test_act, label = 'Actual')
        
        exp_return = 0
        for r in range(config.RANDOM_SAMPLE):

            test_rand = []
            test_rand_idx = []

            random_points = np.random.choice(range(len(test_act)), self.total_buy_points, replace=False) # pick random points from the test interval

            for i in range(len(test_act)):

                if i in random_points:
                    test_rand_idx.append(i)
                    test_rand.append(test_act[i])

                    if i < len(test_act)-1:
                        exp_return += ((test_act[i+1] - test_act[i]) / test_act[i])

            plt.scatter([idx for idx in test_rand_idx], test_rand, label = 'Random buy points {}'.format(r))

        # Set the x & y axis labels
        plt.xlabel('t')
        plt.ylabel('Price ($)')

        # Set a title of the current axes.
        random_return = round(exp_return*100 / config.RANDOM_SAMPLE,2)
        print('Random exp. return: {}%'.format(random_return))
        plt.title('Random buy points (baseline)\nStrategy: selling next day\nExp. return: {}%'.format(random_return))

        # show a legend on the plot
        plt.legend()
        plt.grid()
        plt.savefig('plot_rand.png')
        #plt.show(block=False)


    def writeToFile(self):
        
        df_result = pd.DataFrame(np.array([self.y_test_flat, self.test_pred_flat]).T.tolist(), columns = ['Actual', 'Prediction'])

        df_result.to_excel('stock_price_pred.xlsx')


def main():

    data = Data() # create data object
    data.read() # read stock data

    for i in range(config.NO_RUNS):
        
        rnn = RNN(data.df) # create model object

        rnn.normalizeInput()
        rnn.TrainTestSplit() # prepare dataset for training

        rnn.trainModel() # train model

        rnn.evaluate() # visualize prediction output
        rnn.baseline() # visualise baseline

        rnn.writeToFile() # write predictions to file


if __name__ == "__main__":

    main()




    