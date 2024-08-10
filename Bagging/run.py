#-- train & test the model ---

data = csv_to_numpy('data/DesTreeData.csv')
train, test = train_test_split(data, train_split=.8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

verbose_test = True
n_bootstrap = 100
dtree_dict = {
    'max_depth': 1000,
    'min_node_samples': 2,
    'alpha': 0,
    'verbose_train': False,
    'verbose_test': True,
    'modality': 'gini'
}

model = BaggedTrees(verbose_test = verbose_test) 
model.train(X_train, Y_train, n_bootstrap = n_bootstrap, dtree_dict = dtree_dict)
model.test(X_test, Y_test)

