#!/usr/bin/python3
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def parser_initialization():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='a model file')
    parser.add_argument('x_file', help='a csv file')
    parser.add_argument('output', help='output place')
    args = parser.parse_args()
    return args


def main():
    args = parser_initialization()
    loaded_model = pickle.load(open(args.model, 'rb'))
    inputs = pd.read_csv(args.x_file,thousands=',' , skipinitialspace=True)
    inputs = inputs.drop(['logprice'],axis=1)
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)
    y_hat_test = loaded_model.predict(inputs_scaled)
    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
    df_pf.to_csv(args.output)


if __name__ == "__main__":
    main()