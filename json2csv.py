import pandas as pd

if __name__ == '__main__':
    df = pd.read_json('dataset/stationery.json')
    df.to_csv('dataset/stationery.csv', index=False)

    # codes for adding category column to mobile and tablet group
    # df = pd.read_csv('dataset/mobile.csv')
    # df = df.drop(['category'], axis=1)
    # print(df.columns)
    # # df['category'] = 'tablet'
    # df.insert(2, 'category', 'موبایل')
    # df.to_csv('dataset/mobile_cat.csv', index=False)
