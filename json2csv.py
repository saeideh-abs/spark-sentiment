import pandas as pd


def sentifars_lexicon():
    df = pd.read_csv('./resources/SentiFars_lexicon/SentiFars_lexicon.csv')

    positives = []
    negatives = []
    objectives = []
    df['max'] = df[['positive', 'objective', 'negative']].idxmax(axis=1)
    for index, polarity in enumerate(df['max']):
        if polarity == 'positive':
            positives.append(df.loc[index]['word'])
        elif polarity == 'negative':
            negatives.append(df.loc[index]['word'])
        else:
            objectives.append(df.loc[index]['word'])
    pos = open('positive.txt', 'w')
    [pos.write(elem + '\n') for elem in positives]

    neg = open('negative.txt', 'w')
    [neg.write(elem + '\n') for elem in negatives]

    obj = open('objective.txt', 'w')
    [obj.write(elem + '\n') for elem in objectives]


if __name__ == '__main__':
    # df = pd.read_json('dataset/stationery.json')
    # df.to_csv('dataset/stationery.csv', index=False)

    # codes for adding category column to mobile and tablet group
    # df = pd.read_csv('dataset/mobile.csv')
    # df = df.drop(['category'], axis=1)
    # print(df.columns)
    # # df['category'] = 'tablet'
    # df.insert(2, 'category', 'موبایل')
    # df.to_csv('dataset/mobile_cat.csv', index=False)

    sentifars_lexicon()