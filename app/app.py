import pandas as pd
from catboost import CatBoostClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

model_cb = CatBoostClassifier()
model_cb.load_model("trained_model.cbm")

df = pd.read_csv('online_shoppers_new.csv')
X = df.drop(['Revenue'], axis=1)
y = df['Revenue']
explainer = ClassifierExplainer(model_cb, X.iloc[:500], y.iloc[:500])
db = ExplainerDashboard(explainer)
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
db = ExplainerDashboard.from_config("dashboard.yaml")
db.run(host='0.0.0.0', port=9050, use_waitress=True)