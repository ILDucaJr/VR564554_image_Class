| Method | Trainable params | Time (s) | Train acc | Test acc [95% CI] | Test macro-F1 [95% CI] | Train-test gap |
|---|---|---|---|---|---|---|
| Features + Linear SVM | 21,349 | 49 | 0.999 | 0.833 [0.821, 0.845] | 0.831 [0.818, 0.842] | 0.166 |
| Features + Logistic Regression | 21,349 | 45 | 0.988 | 0.857 [0.846, 0.868] | 0.855 [0.844, 0.865] | 0.131 |
| Features + Random Forest | - | 46 | 1.000 | 0.826 [0.815, 0.838] | 0.822 [0.810, 0.833] | 0.174 |
| Full fine-tuning | 1,555,781 | 100 | 0.998 | 0.856 [0.845, 0.868] | 0.854 [0.843, 0.864] | 0.142 |
| Classifier-head training | 628,773 | 43 | 0.998 | 0.870 [0.860, 0.881] | 0.869 [0.858, 0.878] | 0.127 |
