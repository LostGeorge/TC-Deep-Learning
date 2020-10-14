import random, os
import boto3, s3fs
import numpy as np
import pandas as pd

SAMPLE_FACTOR = 0.1
random.seed(0)
np.random.seed(0)

def sample_images(sample_lst, chn, s3_bucket):
    if not os.path.exists('data_sampled/img'):
        os.makedirs('data_sampled/img')
    if not os.path.exists('data_sampled/img/' + chn):
        os.makedirs('data_sampled/img/' + chn)
    img_info = s3_bucket.objects.filter(Prefix='processed_data/img/' + chn + '/',
                                        Delimiter='/',
                                        MaxKeys=200000).all()
    print(f'{len(img_info)} {chn} image keys retrieved from s3.')
    img_keys = np.array([img.key for img in img_info])
    img_keys.sort()
    img_keys = img_keys[sample_lst]
    for i, k in enumerate(img_keys):
        path = 'data_sampled/img/' + chn + '/' + str(sample_lst[i]).zfill(6) + '.npy'
        s3_bucket.download_file(k, path)
    print(f'{len(img_info)} {chn} images saved to disk.')

def sample_df(sample_lst, df, name):
    df = df.iloc[sample_lst, :]
    df.to_csv('data_sampled/' + name + '.csv')
    print(name + ' saved to disk.')

if not os.path.exists('data_sampled'):
    os.makedirs('data_sampled')

s3 = boto3.resource('s3')
bucket = s3.Bucket('tropical-storms')

wind_df1 = pd.read_csv('s3://tropical-storms/processed_data/wind_data.csv', header=0)
wind_df2 = pd.read_csv('s3://tropical-storms/processed_data/wind_data_2.csv', header=0)
wind_df = pd.concat([wind_df1, wind_df2], axis=0).reset_index(drop=True)
print('Wind data retrieved from s3.')

press_df1 = pd.read_csv('s3://tropical-storms/processed_data/pressure_data.csv', header=0)
press_df2 = pd.read_csv('s3://tropical-storms/processed_data/pressure_data_2.csv', header=0)
press_df = pd.concat([press_df1, press_df2], axis=0).reset_index(drop=True)
print('Pressure data retrieved from s3.')

n_total = len(wind_df)
samples = random.sample(range(n_total), int(n_total * SAMPLE_FACTOR))
samples.sort()

sample_df(samples, wind_df, 'wind_data_sampled')
sample_df(samples, press_df, 'pressure_data_sampled')
sample_images(samples, 'irwin', bucket)
    

