def set_features_and_target(df, target):
    X = df.drop([target])
    y = df.pop([target])
    return X, y
