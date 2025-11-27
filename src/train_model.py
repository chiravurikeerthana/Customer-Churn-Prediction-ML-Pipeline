from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ---------------------------
# Logistic Regression
# ---------------------------
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ---------------------------
# Random Forest
# ---------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# XGBoost
# ---------------------------
def train_xgboost(X_train, y_train):
    imbalance_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=imbalance_weight
    )
    model.fit(X_train, y_train)
    return model
