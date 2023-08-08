import numpy as np
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('online_shoppers_intention.csv')
    print(f'Data consists of {data.shape[1]} columns and {data.shape[0]} rows')

    # make integers from bools
    data['Revenue'] = data['Revenue'].astype(np.int8)
    data['Weekend'] = data['Weekend'].astype(np.int8)

    # replace NaNs with medians
    for col in data.columns[data.isna().sum() > 0]:
        data[col].fillna(data[col].median(), inplace=True)

    # Fix double August in intial dataset
    data['Month'] = data['Month'].replace({'aug': 'Aug'})

    # Feature transfromation
    # Administrative
    data['Administrative_0'] = data['Administrative'].apply(lambda x: int(x == 0))
    data['Administrative_Duration_0'] = data['Administrative_Duration'].apply(lambda x: int(x == 0))
    data['Administrative_Duration_log'] = data['Administrative_Duration'].apply(lambda x: np.log(x) if x > 0 else -1)

    # Informational
    data['Informational_0'] = data['Informational'].apply(lambda x: int(x == 0))
    data['Informational_Duration_0'] = data['Informational_Duration'].apply(lambda x: int(x == 0))
    data['Informational_Duration_log'] = data['Informational_Duration'].apply(lambda x: np.log(x) if x > 0 else -1)

    # ProductRelated
    data['ProductRelated_log'] = data['ProductRelated'].apply(lambda x: np.log(x) if x > 0 else -1)
    data['ProductRelated_Duration_0'] = data['ProductRelated_Duration'].apply(lambda x: int(x == 0))
    data['ProductRelated_Duration_log'] = data['ProductRelated_Duration'].apply(lambda x: np.log(x) if x > 0 else -1)

    # BounceRates
    data['BounceRates_0'] = data['BounceRates'].apply(lambda x: int(x == 0))
    data['BounceRates_log'] = data['BounceRates'].apply(lambda x: np.log(x) if x > 0 else -10)

    # ExitRates
    data['ExitRates_0'] = data['ExitRates'].apply(lambda x: int(x == 0))
    data['ExitRates_log'] = data['ExitRates'].apply(lambda x: np.log(x) if x > 0 else -10)

    # PageValues
    data['PageValues_0'] = data['PageValues'].apply(lambda x: int(x == 0))
    data['PageValues_log'] = data['PageValues'].apply(lambda x: np.log(x) if x > 0 else -1)

    # Drop less important columns
    cols_to_drop = ['Administrative', 'Administrative_Duration', 'Informational',
                    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                    'BounceRates', 'ExitRates', 'PageValues',
                    'Administrative_Duration_0',
                    'Informational_Duration_0',
                    'ProductRelated_Duration_0',
                    'BounceRates_0', 'ExitRates_0', 'PageValues_0',
                    'Region', 'Browser', 'OperatingSystems', 'Administrative_0', 'Informational_0']

    data = data.drop(cols_to_drop, axis=1)

    data.to_csv("online_shoppers_new.csv", index=False)

