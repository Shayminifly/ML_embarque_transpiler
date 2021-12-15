from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd
import joblib
import os
import numpy as np

def trained_linear_regression(X, y):
    lr = LinearRegression().fit(X, y)
    joblib.dump(lr, "model.joblib")
    return lr


def get_coefs(str_model):
    model = joblib.load('model.joblib')

    len_coefs = len(model.coef_) + 1
    coefs = f"{model.intercept_}f,"

    for coef_ in model.coef_:
        coefs += str(coef_) + "f,"
    return len_coefs, coefs.strip(",")



def transpiler(len_coefs, coefs, x_true):
    x_true = str(x_true[0])

    transpiler_lr_code_c = """
    #include <stdio.h>

    float coefs[{0}] = {{{1}}};
    
    float pred_from_coef(float feature)
    {{
        return feature * coefs[1];
    }}
    int main()
    {{
        float X_true = {2};
        printf("%f\\n", pred_from_coef(X_true));
    }}
    """.format(len_coefs, coefs, x_true)

    with open("transpiler_lr.c", "w") as f:
        f.write(transpiler_lr_code_c)


X, y = make_regression(n_features=1, n_informative=1)
sklearn_lr = trained_linear_regression(X, y)
len_coefs, coefs = get_coefs('model.joblib')

X_true = X[3]
y_true = y[3]
transpiler(len_coefs, coefs, X_true)
print(X_true, y_true)
print("y_true:", y_true)
print("Sklearn LR:", sklearn_lr.predict([X_true]))


print("La commande:")
print("gcc transpiler_lr.c -O3 -o main \n./main")
#La commande :
#gcc transpiler_lr.c -O3 -o main
#./main
