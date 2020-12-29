import pandas as pd

if __name__ == '__main__':
    df = pd.read_json('dataset/mobile.json')
    df.to_csv('dataset/mobile.csv', index=False)