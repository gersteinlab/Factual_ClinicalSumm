import argparse
import pandas as pd
import numpy as np
import os

def preprocess_dataset_single(file_path):
    ids, inputs, targets = [], [], []
    is_target = False
    current_input, current_target = '', ''

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()

            if line == '':
                inputs.append(current_input.replace('\t', ' '))
                targets.append(current_target.replace('\t', ' '))
                current_input, current_target = '', ''
                is_target = False
                continue

            if is_target:
                if line.startswith('CONCLUSIONS'):
                    current_target += line[11:] + ' '
                else:
                    current_input += line + ' '
            else:
                if line.startswith('###'):
                    ids.append(line[3:])
                    current_input, current_target = '', ''
                    is_target = True
    
    df = pd.DataFrame()
    df['ID'] = ids
    df['inputs'] = inputs
    df['targets'] = targets
    df['targets'].replace('', np.nan, inplace=True)
    df.dropna(subset=['targets'], inplace=True)
    return df


def preprocess_multi(inputs_df, targets_df):
    review_ids, inputs_de, inputs, targets = [], [], [], []
    for i in range(len(inputs_df)):
        review_id = inputs_df.iloc[i]['ReviewID']
        if review_id not in review_ids:
            review_ids.append(review_id)
            input_de = '<T> ' + inputs_df.iloc[i]['Title']
            input = '<T> ' + inputs_df.iloc[i]['Title']
            if not pd.isna(inputs_df.iloc[i]['Abstract_de']):
                input_de += ' <ABS> ' + inputs_df.iloc[i]['Abstract_de']
                input += ' <ABS> ' + inputs_df.iloc[i]['Abstract']
            inputs_de.append(input_de)
            inputs.append(input)
        else:
            idx = review_ids.index(review_id)
            if not pd.isna(inputs_df.iloc[i]['Abstract_de']):
                inputs_de[idx] += ' <s> <T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract_de']
                inputs[idx] += ' <s> <T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract']
            else:
                inputs_de[idx] += ' <s> <T> ' + inputs_df.iloc[i]['Title']
                inputs[idx] += ' <s> <T> ' + inputs_df.iloc[i]['Title']

    for i in range(len(targets_df)):
        targets.append(targets_df.iloc[i]['Target'])

    for i in range(len(inputs_de)):
        inputs_de[i] = ' '.join(inputs_de[i].replace('\n', ' ').split())

    for i in range(len(inputs)):
        inputs[i] = ' '.join(inputs[i].replace('\n', ' ').split())

    data = pd.DataFrame()
    data['ID'] = review_ids
    data['input_decorated'] = inputs_de
    data['inputs'] = inputs
    data['targets'] = targets
    return data
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--decorated_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'single':
        data_file_list = ['train.txt', 'dev.txt', 'test.txt']
        for data_file in data_file_list:
            data_path = args.dataset_dir + data_file
            df = preprocess_dataset_single(data_path)
            df_path = args.dataset_dir + data_file.split('.')[0] + '.csv'
            df.to_csv(df_path, index=False)

    elif args.mode == 'multi':
        inputs_file_list = ['train_input.csv', 'dev_input.csv', 'test_input.csv']
        targets_file_list = ['train-targets.csv', 'dev-targets.csv', 'test-targets.csv']
        for inputs_file, targets_file in zip(inputs_file_list, targets_file_list):
            inputs_path = args.decorated_dir + inputs_file
            inputs_df = pd.read_csv(inputs_path)
            targets_path = args.decorated_dir + targets_file
            targets_df = pd.read_csv(targets_path)
            data = preprocess_multi(inputs_df, targets_df)
            data_path = args.output_dir + inputs_file.split('_')[0] + '.csv'
            data.to_csv(data_path, index=False)


if __name__ == "__main__":
    main()
