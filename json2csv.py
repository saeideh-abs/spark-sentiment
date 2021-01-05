import pandas as pd

if __name__ == '__main__':
    df = pd.read_json('dataset/headset.json')
    df.to_csv('dataset/headset.csv', index=False)