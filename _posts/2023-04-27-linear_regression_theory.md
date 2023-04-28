---
title: 'Linear Regression: Theory'
date: 2023-04-27
permalink: /notes/linear_regression_theory
tags:
  - cool posts
  - category1
  - category2
---
$
\newcommand{\bx}{\pmb{x}}
\newcommand{\by}{\pmb{y}}
\newcommand{\bX}{\pmb{X}}
\newcommand{\bY}{\pmb{Y}}
\newcommand{\bT}{\pmb{\Theta}}
\newcommand{\bp}{\pmb{\phi}}
\newcommand{\bP}{\pmb{\Phi}}
\newcommand{\bh}{\pmb{h}}
$

In these notes we will introduce the linear regression model, one of the simplest problems to solve in machine learning. Despite its simplicity, it serves as a great starting point to learn many fundamental concepts in the field of machine learning that will reappear several times along the way. In this part of the notes we dedicate purely on the theoretical background. In a [second part](/notes/linear_regression_practice) we will see how all this theory is applied in practice by implementing the model in Python. We will follow closely the references:

* Lecture 4 of **Stanford's CS229 course of 2019:** [video-lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh), [pdf and slides](http://cs229.stanford.edu/syllabus-summer2019.html), and [problem sets](http://cs229.stanford.edu/summer2019/)

* Chapter 3.1 of **Christopher M. Bishop. 2006. Pattern Recognition and machine learning**

## Brief Introduction to Supervised Learning


Before dealing with our particular problem let's see the bigger picture to understand where we are standing. Also, this section serves to establish our conventions which we will use along the whole notes as well as to introduce a lot of common terminology in machine learning. For the latter, it is worth checking out the following [glossary](https://developers.google.com/machine-learning/glossary#labeled_example).

Linear regression is just a particular case among a big class of problems called **supervised learning**, which are also a subset inside **machine learning**. In general, the latter deals with a set of **inputs**, or **features**, $x^{(i)}$ and in supervised learning they come together with **outputs**, or **targets**, $y^{(i)}$. Each pair is packed in a tuple $(x^{(i)},y^{(i)})$ called **training example** and the data set containing the list of all $n$ training examples $\{(x^{(i)},y^{(i)})\}_{i=1}^{N}$ is named **training set**. The latter is what we use to train our model. Denoting, respectively, $\mathbb{X}$ and $\mathbb{Y}$ as the space of inputs and outputs, most of supervised learning algorithms goal is to find a function $h:\mathbb{X} \mapsto \mathbb{Y}$ such that, given our training set, it is a "good" predictor for the outputs. For historical reasons $h$ is usually called a **hypothesis**. Once we fixed this function using the training set, the trained model is ready to be used for making predictions in new data sets that the model has never seen before! In practice, one usually starts from a given data set and split it into training set and **test set**. While the former is used to train the model, the latter is a portion of the data set that does not enter in the training process but it is used to test how well the model performs in new, unknown examples.

Depending on the nature of the output space, supervised learning is divided into **regression** or **classification** problems. The former case corresponds to continuous outputs while in classification problems the targets can only take discrete values. For instance, when $\mathbb{Y} = \{0,1\}$, we have a **binary** problem, in which all outputs satisfy $y^{(i)} \in \{0,1\}$. As you could imagine, linear regression belongs to regression problems, and in what follows we will study this case in detail.

## Linear hypothesis and mean squared error

For the whole problem we will assume $N$ examples such that the inputs belong to a $D$-dimensional real space, $\bx^{(i)} \in \mathbb{R}^D$ and the outputs are real-valued scalars $y^{(i)} \in \mathbb{R}$. For a generic feature $\bx$, we will denote its components as $x_k$ with $k=1, \dots, D$. For generic supervised learning problems $h$ can be any function. **Linear regression** is defined by constraining the hypothesis to be linear in features, i.e.
\begin{equation}\label{eq: LR}
h_{\bT}(\bx) \equiv \Theta_0 + \Theta_1 x_1 + \dots + \Theta_d x_d =\Theta_0+\left(\sum_{k=1}^{D} \Theta_k x_k\right) =\sum_{k=0}^{D} \Theta_k x_k = \bT^{T} \bx\,.
\end{equation}
In the third equality we introduced an **intercept term**, $x_0\equiv 1$, whose only purpose is to allow us to write expressions in a more compact way. The last equality is just the tensor notation for the interior product between two vectors, $\bT$ and $\bx$, both in $\mathbb{R}^{D+1}$.

Here, $\bT$ are called **parameters** or **weights** which parameterize the space of all functions $h_{\bT}$. It is exactly our learning problem to find the parameters that make the hypothesis the "best" predictor. In order to do that we need a way to capture the amount of displeasure between our predicted value $\hat{y} = h_{\bT}(\bx)$ and the right output $y$. This can be achieved defining a **loss function**, also called **cost function** . For this problem we will use the **mean squared error (MSE)**
\begin{equation}\label{eq: MSE}
J(\bT)=\frac12 \sum_{i=1}^{N} \left[h_{\bT}(\bx^{(i)}) - y^{(i)} \right]^2\,.
\end{equation}
Later on we will see why we choose this function! Interestingly, our job of finding the best predictor turned into an optimization problem such that
\begin{equation}\label{eq: Thetahat}
\hat{\bT} \equiv \underset{\bT}{\text{arg min}} \, J(\bT)\,.
\end{equation}
As a side remark, we should mention that, technically speaking, $\eqref{eq: MSE}$ is not exactly MSE but just the squared error function. This is because of the $\frac12$ in front instead of $\frac1N$ which would compute the real "mean" value. However, it is iteresting to notice that in $\eqref{eq: Thetahat}$ the normalization factor of $J(\bT)$ plays no role. The minimal value $\hat{\bT}$ is independent of this overall constant and so we feel free to keep calling $\eqref{eq: MSE}$ MSE. We choose $\frac12$ just for aesthetic reasons that will become clear later.

## Closed form solution

The optimization problems we encounter in machine learning rarely have a closed form solution. However, in linear regression, the minimization problem \eqref{eq: Thetahat} is simple enough to admit an analytic expression. To see this, we first introduce some notation

$$
\bX = 

\begin{pmatrix}\label{XY}
x^{(1)}_{0} & \dots & x^{(1)}_{D}\\
\vdots & \ddots & \vdots\\
x^{(N)}_{0} & \dots & x^{(N)}_{D}
\end{pmatrix} 

\, \in \mathbb{R}^{N\, \times\, D+1}\,, \quad \bY = \left(y^{(1)}, \dots, y^{(N)}\right)^T\, \in \mathbb{R}^{N} \,,
$$

where $X$ is known as **design matrix**. With this notation \eqref{eq: MSE} takes a more succinct form 

\begin{equation}\label{JXY}
J(\bT) = \frac12 \left(\bX \bT - \bY\right)^T\left(\bX \bT - \bY\right) = \frac12 || \bX \bT - \bY ||_2^2\,,
\end{equation}

where in the last equality we recognized the 2-norm. From calculus we know that the extremum of the cost function is defined by the equation $\nabla_{\bT} J(\bT) = 0$. In linear regression the latter gives rise to the **normal equation** and can be solved exactly, leading to the optimal parameter
\begin{equation}\label{eq: exactTheta}
\nabla_{\bT} J(\bT)|_{\bT=\hat{\bT}}=0 \quad \Rightarrow \quad \hat{\bT} = (\bX^T \bX)^{-1} \bX^T \bY\,,
\end{equation}
which requires the matrix $\bX^T\bX$ to be invertible. Since \eqref{eq: MSE} is convex, $\hat{\bT}$ is not only an extremum but the global minimum of $J$.

## Gradient descent

As we already mentioned, we were extremely lucky in finding a closed formula for the optimization problem and this will not happen very often. Because of this, we usually work with numerical optimization algorithms. In particular, \eqref{eq: Thetahat} can be solved with a search algorithm that starts with a random initial value $\bT^{(0)}$ and that iteratively updates its value to decrease $J(\bT)$ until convergence. A specific algorithm of this kind is **gradient descent (GD)** which is defined as follows:

1. Initialize the parameters to random values $\bT^{(0)}$.
1. Update according to the rule $\bT^{(t+1)} = \bT^{(t)} - \alpha \nabla_{\bT} J(\bT^{(t)})$.
1. Repeat step 2 until convergence (given some established criterion).

Steps 2 and 3 can be condensed in the notation
\begin{equation}\label{eq: GD}
\bT := \bT - \alpha \nabla_{\bT} J(\bT)\,,
\end{equation}
where $:=$ stands for "update until convergence". In the algorithm we denoted the gradient as $\nabla$ and we introduced a **learning rate** $\alpha$, which in practice is determined by trying different values and choosing the best. This is an example of a **hyperparameter**, a parameters that we must fix before running the learning algorithm and that cannot be updated or "improved" by the optimization method. In this case, $\alpha$ should not be too large (because we can miss the minimum) or too small (because it would take a lot of time to reach the minimun). It could happen that the exact minimum is never reached by the algorithm so other criteria should be considered. For instance, one can decide to stop when the condition $||\nabla_{\bT} J(\bT^{(t)})||_2 < \epsilon$ is satisfied, for a predetermined $\epsilon$. This is a very natural algorithm that repeatedly takes the step of the steepest decrease of $J(\bT)$!

Up to here the optimization method was totally general, regardless of the form of the cost function. Now, let's move on to our particular case where $J(\bT)$ is the MSE $\eqref{eq: MSE}$ and so its divergence is given by
\begin{equation}
\nabla_{\bT} J(\bT) = \left(\bX \bT - \bY \right)^T \bX\,,
\end{equation}
where now it becomes clear why we choose the prefactor $\frac12$ in \eqref{eq: MSE}. Finally, the update rule reads
\begin{equation}\label{eq: LMS}
\bT:= \bT - \alpha \left(\bX \bT - \bY \right)^T \bX\,.
\end{equation}
This algorithm, which uses GD to solve the linear regression problem, is called **least mean squares (LMS)**. While in general GD can be susceptible to get stuck in a local minima, in linear regression $J(\bT)$ is a *convex quadratic function* and it has only one global minimum, so GD always converges to it!

### Gradient descent variants

In the previous algorithm, in order to make a tiny progress in the direction of the minimum we needed to sum over the *whole* training set (this can be seen from the interior product in $\eqref{eq: LMS}$, which is implicitly a sum from $i=1$ to $N$), which could contain millions or billions examples! In those cases the procedure is very inefficient and other proposals become worth exploring. One of them is a small variant of GD called **stochastic gradient descent (SGD)**. In this case we consider just *one* example at the time $\bx^{(j)}$, which is taken randomly from the training set, and the update rule goes as
\begin{equation}\label{eq: SGD}
\bT:= \bT - \alpha \left[\bT^T \bx^{(j)} - y^{(j)} \right] \bx^{(j)}\,,
\end{equation}
In this case, we "learn" from each example and update the parameters immediately after, whereas previously we had to wait until running over all the examples to make one single change in $\bT$. The idea is to run $\eqref{eq: SGD}$ for all examples in the training set, and then repeat all over again until convergence. Often, SGD gets closer to the minimum faster than GD (albeit taking more steps). However, since at each step we are not using the information of the entire cost function, but just the one produced by a single example, SGD never finds exactly the minimum but oscillates around it, getting closer and closer. Finally, there is another intermediate variation between GD and SGD named **mini-batch SGD** which takes a small random sample of examples (mini-batch) to update the parameters
\begin{equation}\label{eq: mini-batch_SGD}
\bT:= \bT - \alpha \sum_{i=1}^{B} \left[\bT^T \bx^{(i)} - y^{(i)} \right] \bx^{(i)}\,,
\end{equation}
where $1< B < N$ is the length of the batch. As in SGD, we perform $\eqref{eq: mini-batch_SGD}$ until exhausting all examples and then repeat the algorithm until convegence.  For all these optimization algorithms, we say that we completed and **epoch** of training each time we went trhough the whole training set. In general, to train a given model one requires to repeat the algorithm for several epochs. As a side remark, the batch length and the number of epochs are two more examples of hyperparameters.



## Probabilistic interpretation

In the previous section, by using the linear hypothesis in $\eqref{eq: LR}$, we assumed that we could approximate the outputs by a linear combination of the inputs. Also, in $\eqref{eq: MSE}$ we decided to use MSE as the cost function to measure the model's performance. These definitions lead to the LMS algorithm $\eqref{eq: LMS}$. Now, one could ask why we made such assumptions. In this section we will motivate LMS as a natural algorithm emerging from some fundamental probabilistic assumptions on the model.

The linear hypothesis $\eqref{eq: LR}$ can be rephrase it in a probablistic language by proposing that the outputs are a linear function of the inputs *plus a random Gaussian noise*
\begin{equation}
y^{(i)} = \bT^T \bx^{(i)} + \epsilon^{(i)}\,, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)\,,
\end{equation}
where $ \mathcal{N}(0, \sigma^2)$ is a normal distribution with mean zero and variance $\sigma^2$. This implies that $y^{(i)}$ is also a random variable following the distribution
\begin{equation}
y^{(i)} \sim  \mathcal{N}(\bT^T \bx^{(i)}, \sigma^2)\,,
\end{equation}
which can be also expressed as
\begin{equation}\label{eq: pyx}
P(y^{(i)} |\, \bx^{(i)} ; \bT) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{1}{2 \sigma^2} \left(y^{(i)} - \bT^T \bx^{(i)}\right)^2}\,,
\end{equation}
which introduces the conditional probability distribution of $y^{(i)}$ given $\bx^{(i)}$ and parameters $\bT$. Considering all the examples together we have instead $P(\bY | \bX ; \bT)$. This quantity is viewed as a function of the examples for a fixed value of $\bT$. However, the very same distribution can be interpreted as a function of the parameters instead, in which case 
\begin{equation}
L(\bT) = L(\bT; \bX, \bY) \equiv P(\bY | \bX ; \bT)\,,
\end{equation}
receives the name of **likelihood function**. By assuming that all the examples are **independently and identically distributed (iid)** we can separate the likelihood in products of examples, i.e.
\begin{equation}
L(\bT) = \prod_{i=1}^{N} P(y^{(i)} | \bx^{(i)}; \bT)\,.
\end{equation}
In this new interpretation of the problem, the optimum parameters $\hat{\bT}$ are determined by assumming the principal of **maximum likelihood** which states
\begin{equation}
\hat{\bT} = \underset{\bT}{\text{arg max}} \, L(\bT) = \underset{\bT}{\text{arg max}} \, \ell(\bT)\,, \quad \ell(\bT) \equiv \log L(\bT)\,,
\end{equation}
where in the last equality we introduced the **log likelihood function** which gives the same optimal parameter as $L(\bT)$ because the logarithm is a strictly increasing function. This is a trick that allows us to replace products by sums, a major simplification that becomes handy for the optimization problem. By using $\eqref{eq: pyx}$ in the definition of $\ell(\bT)$ we can see that 
\begin{equation}
\ell(\bT) = N \log \frac{1}{\sqrt{2 \pi \sigma^2}} - \frac{1}{\sigma^2} \frac12 \sum_{i=1}^{N} \left(y^{(i)} -\bT^T \bx^{(i)}\right)^2\,.
\end{equation}
Hence, maximazing $\ell(\bT)$ gives exactly the same result as minimazing MSE $\eqref{eq: MSE}$! 
\begin{equation}\label{eq: lJ}
\hat{\bT} = \underset{\bT}{\text{arg max}} \, \ell(\bT) = \underset{\bT}{\text{arg min}} \, J(\bT)\,.
\end{equation}

To sum up, under the assumption of a linear dependency between input and outputs of idd examples together with a Gaussian noise, the maximum likelihood principle gave us the same answer for the optimal parameters than LMS. This is just a particular case of a more general statement which reads also like $\eqref{eq: lJ}$ relating negative log likelihood functions with loss functions.

## Regularization

Since linear regression is one of the simplest problems in machine learning, it is a good point to introduce the concept of **regularization**. The general ideas developed here will also apply to other more complex problems. 

Regularization is a method to prevent over-fitting, which can occur, for instance, when our model has many degrees of freedom compared to the number of training examples. In these scenarios, the model "memorizes" the training set and fails to generalize to new data. There are many regularization schemes, here we will cover one that modifies the cost function \eqref{JXY} by adding a new term of the form $\lambda J_R(\bT)$, where $\lambda$ is a **regularization coefficient** that controls the relative importance of the data-drive cost function and the regularizer. This is another example of a hyperparameter. One of the simplest form for $J_R(\bT)$ is given by
\begin{equation}
J_R(\bT) = \frac12 \bT^T \bT = \frac12 ||\bT||_2^2\,,
\end{equation}
and so the regularized cost function takes the form
\begin{equation}
J(\bT) = \frac12 || \bX \bT - \bY ||_2^2\, +\, \frac{\lambda}{2} ||\bT||_2^2\,.
\end{equation}
This particular regularization choice is called **weight decay** because it prevents the parameters from reaching large values. Against other regularizers, it has the advantage that the cost function remains quadratic in weights so a closed form for its minimum is still possible. The vanishing of the gradient w.r.t $\bT$ gives the analogous of \eqref{eq: exactTheta}
\begin{equation}
\hat{\bT} = (\bX^T \bX + \lambda \pmb{I})^{-1} \bX^T \bY\,,
\end{equation}
where $\pmb{I}$ is the $(D+1) \times (D+1)$ identity matrix. 

More general regularizers can be obtained by considering the $q$ norm instead, namely 

$$
J(\bT) = \frac12 || \bX \bT - \bY ||_2^2\, +\, \frac{\lambda}{2} ||\bT||_{q}^{q}\,,
$$

where $$\| {\bT} \|{}_{q}^{q} = \sum_{k=0}^{D} \| \Theta_{k} \|^{q}\,.$$ In particular, the case of $q = 1$ is know as **lasso** regularization and it has the property that, for large $\lambda$, it drives some parameters $\Theta_{j}$ to zero, leading to a sparse model.



## Extensions

Since we already did all the hard work for the simplest linear regression model \eqref{eq: LR}, it is worth considering some generalizations of the problem where our previous results can be extended in a straight-forward manner

### Basis functions

All the simplicity of \eqref{eq: LR} comes from the model being linear in the parameters, not in features. The latter condition can be easily relaxed while keeping the former. To do so we introduce **basis functions** $\phi_j(\bx)\,, j=0, \dots, M$ with the intercept term $\phi_0(\bx) = 1$, and define the corresponding hypothesis
\begin{equation}\label{linear}
h_{\bT}(\bx) \equiv \sum_{k=0}^{M} \Theta_k \phi_k(\bx) = \bT^T \bp(\bx)\,.
\end{equation}
This extension allow us to introduce non-linearities in the features entering trough the non-linear functions $\bp$. While the domain is the $D+1$-dimensional space of inputs, the image of such functions has dimension $M$ which can be smaller or bigger than $D$. As a consequence, in this model the total number of parameters is $M+1$. Hypothesis of the form \eqref{linear} define **linear models** and linear regression \eqref{eq: LR} is just a particular case where $M=D$ and the basis functions are the identity map $\phi_j(\bx) = x_j$. More interesting basis functions are polynomials, Gaussian, sigmoidal, etc. Whichever non-linear function we use, all previous results for linear regression are easily translated to linear models by the naive replacement of $x_j^{(i)} \rightarrow \bp_j(\bx^{(i)})$! For instance, introducing the analogous to the design matrix \eqref{XY} for $\bp$

$$
\bP \equiv \begin{pmatrix}
\phi_0(\bx^{(1)}) & \dots & \phi_M(\bx^{(1)})\\
\vdots & \ddots & \vdots\\
\phi_0(\bx^{(N)}) & \dots & \phi_M(\bx^{(N)})\\
\end{pmatrix}\, \in \mathbb{R}^{N \times M+1}\,,
$$

the cost function \eqref{JXY} now reads
\begin{equation}
J(\bT) = \frac12 ||\bP \bT - \bY||_2^2\,,
\end{equation}
while the update rule for LMS \eqref{eq: LMS} takes the form
\begin{equation}\label{eq: LMSp}
\bT:= \bT - \alpha \left(\bP \bT - \bY \right)^T \bP\,.
\end{equation}

### Multiple outputs

So far we have considered the case where the output examples are one-dimensional, namely $y^{(i)} \in \mathbb{R}$. However, the extension to higher-dimensional outputs is really straightforward, mostly because our previous results were written in tensor notation. For instance, considering $K+1$-dimensional examples $\by^{(i)} = \left(y_0^{(i)}, \dots, y_K^{(i)}\right)^T$, we can extend the vector of parameters $\bT \in \mathbb{R}^{M+1}$ to a matrix $\bT \in \mathbb{R}^{M+1 \times K + 1}$ such that the new hypothesis function is given by
\begin{equation}\label{hpy}
\bh_{\bT}(\bx) = \bT^T \bp(\bx)\, \in \mathbb{R}^{K+1}\,,
\end{equation}
which in components reads
\begin{equation}
h_{\bT j}(\bx) = \sum_{k=0}^{M} \Theta_{k j} \phi_k(\bx)\,.
\end{equation}
Another approach would be to define different basis functions for each output component, namely sending $\phi_k(\bx) \rightarrow \phi_{k j}(\bx)$. This would lead to multiple ($K+1$), decoupled linear models, which is not so interesting. \eqref{hpy} is the most common approach to tackle the multiple output problem.

With this new hypothesis, all previous results written in tensor notation hold without changes after sending the output vector $\bY$ from \eqref{XY} to a "design matrix", namely

$$
\bY \equiv \begin{pmatrix}
y^{(1)}_{0} & \dots & y^{(1)}_{K}\\
\vdots & \ddots & \vdots\\
y^{(N)}_{0} & \dots & y^{(N)}_{K}
\end{pmatrix} \in \mathbb{R}^{N \times K + 1}\,.
$$
