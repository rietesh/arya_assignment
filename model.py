from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import ExtraTreeClassifier

class Model:
    def __init__(self, n_features, clfs) -> None:
        self.clfs = clfs
        self.n_features = n_features
        if isinstance(self.clfs, list):

            assert len(clfs)==8, "Provide 8 estimators"

            self.clf1 = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=n_features)),
            ('scale', StandardScaler()),
            ('classification', clfs[0])
            ])

            self.clf2 = Pipeline([
            ('feature_selection', SelectKBest(mutual_info_classif, k=n_features)),
            ('scale', StandardScaler()),
            ('classification', clfs[1])
            ])

            self.clf3 = Pipeline([
                ('scale', MinMaxScaler()),
                ('feature_selection', RFE(LogisticRegression(),n_features_to_select=n_features, step=1)),
                ('classification', clfs[2])
            ])

            self.clf4 = Pipeline([
                ('scale', MinMaxScaler()),
                ('feature_selection', RFE(LinearSVC(),n_features_to_select=n_features, step=1)),
                ('classification', clfs[3])
            ])

            self.clf5 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(KNeighborsClassifier(weights='distance'), n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs[4])
            ])

            self.clf6 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs[5])
            ])

            self.clf7 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(ExtraTreeClassifier(), direction='forward',n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs[6])
            ])

            self.clf8 = Pipeline([
                ('feature_selection', PCA(n_components=n_features)),
                ('scale', StandardScaler()),
                ('classification', clfs[7])
            ])

        else:
            self.clf1 = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=n_features)),
            ('scale', StandardScaler()),
            ('classification', clfs)
            ])

            self.clf2 = Pipeline([
            ('feature_selection', SelectKBest(mutual_info_classif, k=n_features)),
            ('scale', StandardScaler()),
            ('classification', clfs)
            ])

            self.clf3 = Pipeline([
                ('scale', MinMaxScaler()),
                ('feature_selection', RFE(LogisticRegression(),n_features_to_select=n_features, step=1)),
                ('classification', clfs)
            ])

            self.clf4 = Pipeline([
                ('scale', MinMaxScaler()),
                ('feature_selection', RFE(LinearSVC(),n_features_to_select=n_features, step=1)),
                ('classification', clfs)
            ])

            self.clf5 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(KNeighborsClassifier(weights='distance'), n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs)
            ])

            self.clf6 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs)
            ])

            self.clf7 = Pipeline([
                ('feature_selection', SequentialFeatureSelector(ExtraTreeClassifier(), direction='forward',n_features_to_select=n_features, n_jobs=-1, cv=2, scoring='accuracy')),
                ('scale', StandardScaler()),
                ('classification', clfs)
            ])

            self.clf8 = Pipeline([
                ('feature_selection', PCA(n_components=n_features)),
                ('classification', clfs)
            ])

    def fit(self, X, y):
        self.classifiers = [self.clf1, self.clf2, self.clf3, self.clf4, self.clf5, self.clf6, self.clf7, self.clf8]
        self.clf = VotingClassifier(estimators=[('clf'+str(i), self.classifiers[i]) for i in range(len(self.classifiers))],voting='soft')
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]
        


y_train_proba = clf.predict_proba(X_train)
y_test_proba = clf.predict_proba(X_test)

precision_t, recall_t, thresholds_t = precision_recall_curve(y_train,  y_train_proba[:,1])
precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba[:,1])
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
print('Best Threshold {}'.format(thresholds[ix]))

f = plt.figure(figsize=(15,10))
plt.plot(thresholds, precision[:-1], "b--", label="Precision")
plt.plot(thresholds, recall[:-1], "g--", label="Recall")
plt.axvline(x=thresholds[ix])
plt.xlabel("Threshold")
plt.title('Test: Preecision, recall vs Thresholds')
plt.legend()
plt.show()
