#!/usr/bin/python3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import pickle
import numpy as np


def parser_initialization():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('x_file', help='a csv file')
    parser.add_argument('y_file', help='a csv file')
    parser.add_argument('output', help='output place')
    args = parser.parse_args()
    return args


def process_input(x, y):
    raw_data = pd.read_csv('X_train.csv',thousands=',' , skipinitialspace=True)
    raw_data1 = pd.read_csv('Y_train.csv',thousands=',' , skipinitialspace=True)
    raw_data['price'] = raw_data1['price']
    data = raw_data.drop(['model'],axis=1 )

    data= data.dropna(axis=0)
    log_price = np.log(data['price'])
    data['logprice'] = log_price

    data_cleared = data.drop(['odometer'],axis =1 )
    data_cleared = data_cleared .drop(['engineType'],axis =1 )
    data_cleared = data_cleared .drop(['engineCapacity'],axis =1 )
    data_cleared = data_cleared .drop(['photos'],axis =1 )

    data_dummies = pd.get_dummies(data, drop_first=True)
    data_dummies= data_dummies.astype(float)

    cols = data_dummies.columns.values
    data_preprocessed = data_dummies[cols]
    data_preprocessed['logprice'] = np.log(data_preprocessed['price'])
    data_preprocessed=data_preprocessed.drop(['price'],axis=1)

    targets = data_preprocessed['logprice']
    inputs = data_preprocessed.drop(['logprice'],axis=1)

    return inputs, targets


def train(inputs,targets, filename):
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)
    x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size=0.2,random_state=365)
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    pickle.dump(reg, open(filename, 'wb'))
    return


def main():
    args = parser_initialization()
    inputs, targets = process_input(args.x_file, args.y_file)
    train(inputs,targets, args.output)


if __name__ == "__main__":
    main()