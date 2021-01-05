import pandas as pd

if __name__ == '__main__':
    df = pd.read_json('dataset/camera.json')
    df.to_csv('dataset/camera.csv', index=False)