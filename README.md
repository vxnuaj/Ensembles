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

**This** average of multiple classifiers is an **ensemble** of classifiers, which reduces the $h_D(x) - \bar{h(x)}^2$ or the $Var$ of the model output. This can be drawn from drawing multiple bootstrapped datsets, $d_i$ from the overarching dataset $D$.

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

## Boosting 

### Adaboost

Boosting adds a set of weights, to a given set of samples, to iteratively train a set of ensembled trees, allowing them to pay more attention to classes that they've incorrectly classify in previous iterations.

Given a set of training examples, you

1. Train a weak learner, meaning a model that is only slightly better than random guessing.
2. Identify the misclassified and correctly classified classes. 
3. Increase the weights for the samples in the misclassified class (can be done via increaisng probability of drawing a class in bootstrapping)
4. Retrain the model on the new set of samples.
5. For a final prediction, run a pass through all the weak learners and 

Say we have a boosted classifier:
$H(\vec{x}) = \sum_{i=1}^T \alpha_t h_t(\vec{x})$

$h_t$ is an instance of a **weak learner**, a model / algorithm that is not good at classifying predictions.

This is typically a tree stump, a decision tree that has a depth $< 1$.

The weak learner, $h_t$, is trained on a given dataset, only allowed to reach $depth = 1$. The total error for the stump is the $\sum_{i:h(x_i)≠y_i} w_i$, where $w_i$ is the $ith$ weight associated with the $ith$ classifier that provided an incorrect prediction.

This total error will always be within the range $[0, 1]$, as the weights are always normalized to sum up to $1$.

This total error ($\epsilon$) will determine the amount of say or contribution, $\alpha$, that a model has on the final output of the ensemble.

The $\alpha$ is computed as:

$\alpha = \frac{1}{2} ln(\frac{1 - \epsilon}{\epsilon})$

When the total error is small, the amount of say will be large and positive. Otherwise it will be a large negative value.

Then, if $\alpha$ is a negative value, the learner's predictions are incorrect and what would've been, for example, a prediction of $1$ for a positive value, will be turnt into a value representing the opposite class, typically $-1$ or $0$.

As an example, if the total error was defined as $\frac{3}{8}$ or $.625$, the amount of say would be calculated as:

$\alpha = ln (\frac{(1-.625)}{.625}) * (\frac{1}{2})$

For every incorrect mapping that each $h_t$ applies on $\vec{x} \rightarrow \vec{y}$, where $\vec{y}$ are the true labels, you take the initial weak learner $h_1$, and iteratively update the weights on the set of samples, such that the learner $h_1$ pays more attention to the weighted samples at it's second iteration. 

This can be done through a loss function, denoted as:

$l = \sum_{i=1}^ne^{-yh(x_i)_i}$

Once the loss function is computed, the weight update can be computed as:

$w \leftarrow w(e^{-\vec{\alpha} \vec{h(x_i)}y_i})$

*This equation can be eperated into 2 equations as:*

*If correct: $w \leftarrow w(e^{-\vec{\alpha}})$* <br>
*Else: $w \leftarrow w(e^{\vec{\alpha}})$*

*The first more explicitly computes the full weight increase / decrease operation in a single equation while the latter does it seperately.*

where $y_i$ is the true label for the training sample $x_i$ and $\alpha$ is the amount of say that each ensemble, $\vec{h(x_i)}$ is the label prediction from the weak learners.

Then the weights, $w$, are normalized such that the sum of all weights $w$ add up to $1$. 

This can be done by: 

$w_{sum} = \sum w$ <br>
$w_{normalized} = \frac{w}{w_{sum}}$ <br>
$w \leftarrow w_{normalized}$

Then we train another instance of the algorithm, $h_2$, applying the weights $w$. This can be done via weighted bootstrapping or a weighted gini index / entropy.

- The weighted gini index would look as $1 - \sum w^2$, replacing $w$ with the original probability $p$.
- Taking a new dataset, based on weighted bootstrapping, would just increase the probability that a given sample is drawn based on the weights. You draw a new dataset baesd on the weights, but then reset the weights each time, allowing the weak learner to generate new weights on the weighted boostrapped dataset.

The final model consists of all weak learners, with the updated weights, attained at the final iteration of the training. 

Each weak learner then makes a final prediction, based on their amount of say, contributing to the final ensemble as:

$H(\vec{x}) = \sum\alpha h(\vec{t})$

If $H(\vec{x}) > 0$, the sample is classified as $1$, otherwise the sample is classified as $0$ or $-1$, depending on what the opposing label is identified as.


>*Thank you Josh Starmer & Cornell University 🎔*