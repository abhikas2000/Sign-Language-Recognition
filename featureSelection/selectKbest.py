from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd



df=pd.read_csv('featureSelection/data.csv')
X = df.drop(columns=['Target']).values

y = df['Target'].values

selector = SelectKBest(score_func=f_classif, k=20) 


X_new = selector.fit_transform(X, y)


selected_feature_indices = selector.get_support(indices=True)

print("Selected feature indices:", selected_feature_indices)

for i in selected_feature_indices:
    print(df.columns[i],end=',')
print()


selected_feature_scores = selector.scores_[selected_feature_indices]
print("Scores of selected features:", selected_feature_scores)
