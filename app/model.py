import pandas as pd
import numpy as np
from catboost import CatBoostClassifier # Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

if __name__ == "__main__":
    data = pd.read_csv("online_shoppers_new.csv")

    X = data.drop('Revenue', axis=1)
    y = data['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pos_scale = y_train[y_train == 0].shape[0] / y_train[y_train > 0].shape[0]
    categorical_features_indices = np.where(X.dtypes == object)[0]

    params = {'n_estimators': 109, 'max_depth': 3, 'learning_rate': 0.055}

    model = CatBoostClassifier(**params,
                               cat_features=categorical_features_indices,
                               scale_pos_weight=pos_scale,
                               logging_level='Silent')

    model.fit(X_train,
              y_train,
              eval_set=(X_test, y_test))
    y_pred = model.predict(X_test)

    print('roc_auc_score           = ', roc_auc_score(y_test, y_pred))
    print('f1_score                = ', f1_score(y_test, y_pred))
    print('accuracy_score          = ', accuracy_score(y_test, y_pred))

    # model.fit(X, y)
    model.save_model("trained_model.cbm")
