# Authors: Hamza Tazi Bouardi & Pierre-Henri Ramirez
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

train_data = pd.read_csv("data/adult_train.csv")
test_data = pd.read_csv("data/adult_test.csv")

X_train = train_data.iloc[:,:-1].values
Y_train = train_data.iloc[:,-1].values

X_test = test_data.iloc[:,:-1].values
Y_test = test_data.iloc[:,-1].values

def plot_ROC(preds_prob, mod_name, Y=Y_test):
    fpr, tpr, threshold = roc_curve(Y, preds_prob)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve - ' + mod_name)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

#Params
params_logr = {
    "penalty": ["l1", "l2"],
    "C" : [.01,.1,1,10]
}
params_elnet_logr = {
    "penalty": ["elasticnet"],
    "solver": ["saga"],
    "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}
cols_res = (['params'] + ["split"+str(k)+'_test_score' for k in range(10)]+
           ['mean_test_score', 'std_test_score', 'rank_test_score']
)

# We will perform L1 and L2 regularizations
logreg_model = LogisticRegression(random_state=42)
cv_reg_logreg = GridSearchCV(logreg_model, params_logr, cv=10, n_jobs=-1, scoring="roc_auc")
cv_reg_logreg.fit(X_train, Y_train)
results_logreg = pd.DataFrame(cv_reg_logreg.cv_results_)[cols_res]

# Now both at the same time (Elastic Net)
logreg_model = LogisticRegression(random_state=42)
cv_elnet_logreg = GridSearchCV(logreg_model, params_elnet_logr, cv=10, n_jobs=-1, scoring="roc_auc")
cv_elnet_logreg.fit(X_train, Y_train)
results_elnet_logreg = pd.DataFrame(cv_elnet_logreg.cv_results_)[cols_res]
results_logreg = pd.concat([results_logreg, results_elnet_logreg], axis=0)
results_logreg["rank_test_score"] = results_logreg.mean_test_score.rank(method="max", ascending=False)
print(results_logreg)

# Selecting best model and plotting ROC Curve
best_model_logreg = LogisticRegression(C=.1, penalty="l1", random_state=42)
best_model_logreg.fit(X_train, Y_train)
preds_logreg = best_model_logreg.predict(X_test)

preds_train_logreg_prob = best_model_logreg.predict_proba(X_train)[:, 1]
preds_logreg_prob = best_model_logreg.predict_proba(X_test)[:, 1]

acc_logreg = accuracy_score(Y_test, preds_logreg)
auc_logreg = roc_auc_score(Y_test, preds_logreg_prob)
auc_train_logreg = roc_auc_score(Y_train, preds_train_logreg_prob)
print("Logistic Regression test scores \n "+
      f"Accuracy={round(acc_logreg,3)}\n AUC={round(auc_logreg,2)}\n"+
     "Train scores :\n" + f" AUC={round(auc_train_logreg,2)}")

# Plotting the ROC curve
plot_ROC(preds_logreg_prob, "Logistic Regression")
