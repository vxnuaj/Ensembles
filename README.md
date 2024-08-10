# Ensembles

> [!NOTE]
>
> Check out the implementations in the respective folders.

1. [Notes](README.md)
2. [Bagging Implementation](Bagging/bagging.py)
3. [Random Forest & Extremely Randomized Trees implementations](RandomForest/randomforest.py)

## Bagging

**Bagging**, also known as bootstrap aggregating, is an Ensemble Learning method, that uses the same learning algorithm for all $n$ models within the Ensemble.

It uses $n$ Bootstrap Samples to train $n$ total models.

***Bootstrapped samples*** are a set of samples $\hat{X}$ that are drawn from an overarching set of samples $X$ through a uniform distribution.

$\hat{X}$ has samples of $X$ but may miss some samples of $X$ and instead repeat some samples of $X$ multiple times.

This is called sampling with replacement.

The total size of $\hat{X}$ is the same size as $X$.

Then, from each prediction from a given model, $f_i$, you can use the hard majority vote or the soft majority vote from Majority Voting Classifiers and Soft Voting Classifiers respectively, to compute the final prediction for the ensemble.

Given this, in Bagging, it is practical to overfit numerous models on different sets of Bootstrap Samples, and then take the $argmax()$ of a set of averaged probabilities for a set of classes to get a more precise prediction.

> *Bagging is more geared for decision trees, so if you train multiple decision trees without pruning on different Bootstrap Samples, to overfit, and then take the hard or soft majority you'd hypothetically get a higher accuracy.*

If the error of a model is computed as:

$Error = Var + Bias + \epsilon$, where $\epsilon$ is the irreducible error, the goal of bagging is to reduce $Var$.

If $Var = (h_D(x) - \bar{h(x)})^2$, meaning $\sigma^2$, our goal would be to reduce $h_D(x) - \bar{h(x)})^2$. Reducing it would mean that we're introducing a more generalizable model to solve different problems.

>*Simple bias / variance problem, where we want to reduce variability but also mitigate the bias as much as possible, for both accuracy and generalizability.*

**Weak Law of Large Numbers (WLLN)** states that for a sequence of i.i.d (independent and identically distributed) random variables, say $x_i$, with a common mean, $\bar{x}$, we have $\frac{1}{m}\sum_{i=1}^m x_i \rightarrow \bar{x} as m \rightarrow \infty$

So for a hypothetically infinite set of i.i.d datapoints, $x_i$'s, the average of it will return to the same $\bar{x}$

If the **WLLN** is applied to classifiers, a set of bagged classifiers, where the output of an individual classifier $f_i$ is $\hat{y}_i$, then the more classifiers we have, the more representative the averaged output, as $\bar{\hat{y}}$, will be to the true label $y$.

This average of multiple classifiers is an **ensemble** of classifiers, which reduces the $h_D(x) - \bar{h(x)}^2$ or the $Var$ of the model output. This can be drawn from drawing multiple bootstrapped datsets, $d_i$ from the overarching dataset $D$.

So say we have the datsaet $D$ and we want to draw subsets of the data, $d_i$, from $D$, uniformly. The probability distribution that a given $(x_i, y_i)$ be chosen from $D$ can then be denoted as $Q((X, Y) | D)$, where the probability of choosing a given $(x_i, y_i)$ pair is $Q((x_i, y_i) | D) = \frac{1}{n}$ for all $(x_i, y_i) \in D$, where $n$ is equal to the size of $D$.

- *Each sample has an equivalent probability of being chosen for the given subset of $D$, $d_i$*
- *Note that each sample can be chosen more than once, as for each draw, we're drawing from the entire dataset $D$. This is called drawing with replacement.*

Then, the bagged classifier can be denoted as $h_D = \frac{1}{m} \sum_{i = 1}^{m} h_{d_i}$, where $h_D$ is the output of the bagged ensemble, $d_i$ si the subset of samples, $m$ is the total amount of  classifiers, in the bagged ensemble.

- *Note that the **WLNN** does not apply to a bagged classifiers as the subsets, $d_i$ are drawn from $D$ in a manner that doesn't allow for every $d_i$ to be i.i.d as multiple samples can repeat and be dependent. But this does not disrupt the classifier as it still tends to be more empirically accurate than standalone decision trees.*

An advantage of a bagged classifier is that it can provide us with an out-of-the-box test error. Given that some $d_i$ won't include some $(x_i, y_i)$, there will be a set of classifiers, $h$, within the total set $H$, that were never trained on $(x_i, y_i)$

Therefore, what one can do is identify the classifiers that weren't trained on $(x_i, y_i)$, and run a prediction using the $(x_i, y_i)$ for that given subset of classifiers. This is run for all classifiers within an ensemble that weren't trained on a given $(x_i, y_i)$ and then ran for all possible sets of $(x_i, y_i)$. The subset of classifiers will differ for each $(x_i, y_i)$. Then the error is averaged amongst all classifiers.

$E = \frac{1}{n} \sum_{(x_i, y_i) \in D} e$

This can then give us an insight on what the true test error would be if the model was implemented on a real-world dataset, without having access to one.

We can also obtain the $\mu$ and the $Var$ for the entire set of classifiers, the $\mu$ being the prediction of the ensemble baesd on soft or hard majority voting and the $Var$ being the level of uncertainty, of the predictions of the model.

## Random Forests

**Random forests** are a set of bagged decision trees, with the addition of some extra randomness to each tree.

Rather than only using Bootstrap Samples, a random forest has it's individual bagged decision trees trained on a random subset of features.

If we have $n$ total features, we choose $k$ features randomly at each node, where $k < n$, limiting a decision tree to choose the optimal feature split index from only the subset of features, $k$.

Then per usual, you'd choose the split with the highest Information Gain through the Gini Index or Entropy, to make the next split, but only based on $k$.

A typical choice of the size of $k$ is $k_{size} = \sqrt{n}$

This is done to decorrelate each tree within the ensemble, to further reduce the variance and mitigate overfitting. 

- *Otherwise, an individual tree in the ensemble might still come up with similar splits to others.*

If we had a dataset $D$, containing multiple $D_i$, where $D_1$ was a very strong feature / predictor for a given label, most trees if not all would then use $D_1$ at the first split, the root split. Thereby most of the trees may end up looking very alike given that the strongly correlated feature was used as the root split node for all.
So each tree is correlated. Random forests overcome this as they're limited to using a subset of the total features / predictors and therefore, most of the trees in the random forest ensemble, won't have a predictor that strongly correlates the trees.

The smaller the size of the subset $D_1$ is, the better the tree will be at generalizing to a dataset to provide an unbiased prediction, as the predictions are based on trees using different sets of predictors.

The process of training a Random Forest is very similar to training a set of bagged trees, the only difference being that we select different $n$ subsets of bootstrapped samples containing different $m$ features $\in D$, $D_i$, to train $n$ different models.

1. Sample $n$ datasets from $D$ with replacement (bootstrapped samples)
2. For each subset, $D_i$, select a random number of features at each node, $m$, to train a decision tree on and leave out the rest, where $m ≤ len(D_i)$
3. Train the ensemble on each $D_i$, and then run predictions
4. Get the final predictions using hard or soft majority voting.

### Extremely Randomized Trees

Similar to random forests, but the subset of trees have more variance in their predictions, each tree is even more different than each.

While the Random Forest makes use of bootstrapping and random feature subsets at each node, Extremely Randomized Trees select a random feature **split** at each node, which increases the variability for each split, given that the **feature split** values vary per split randomly.

The hyperparameters we can change are the:

- The number of bootstrapped samples
- The number of Random Features to consider at each node.
- The number of Random Feature Splits (Threshold values) to consider
