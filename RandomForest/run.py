data = csv_to_numpy('data/DesTreeData.csv')
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last') # dataset has 9 features

verbose_train = True
n_bootstraps = 5
rtree_dict = {
        'verbose_train': False,
        'min_node_samples': 2,
        'max_depth': 100, 
        'criterion': 'entropy'
}

model = RandomForest()
model.train(X_train, Y_train, max_features = 5, n_bootstraps = 5, rtree_dict = rtree_dict)
model.test(X_test, Y_test)

