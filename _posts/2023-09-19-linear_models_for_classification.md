---
title: 'Linear Models for Classification'
date: 2023-09-19
permalink: /notes/linear_models_for_classification
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
\newcommand{\bTe}{\pmb{T}}
\newcommand{\bp}{\pmb{\phi}}
\newcommand{\bP}{\pmb{\Phi}}
\newcommand{\bh}{\pmb{h}}
$

In these notes we cover linear models for solving classification
problems in machine learning. After describing some general
features, we present the **logistic regression** model for binary
classification. We will take this opportunity to introduce a new
optimization algorithm, the **Newton-Raphson** method. We then move up
to a multiclass extension of logistic regression, **softmax
regression**.

Good references for these topics are:

-   Lecture 5 of **Stanford's CS229 course of 2019**:
    [video-lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh),
    [pdf and slides](http://cs229.stanford.edu/syllabus-summer2019.html)
    and [problem sets](http://cs229.stanford.edu/summer2019/)

-   Chapter 4 of **Christopher M. Bishop. 2006. Pattern Recognition and
    machine learning**.

## General characteristics of classification models

Classification models aim to solve the problem of assigning a given
$D$-dimensional input vector $\pmb{x}$ to one of $K$ discrete classes
$C_k, k=1, \dots, K$. This is achieved by separating the input space
into regions via the so-called **decision boundaries**. When this limit
surfaces are $D -1$ hyper-planes, meaning that the boundaries are
defined by linear functions of $\pmb{x}$, the models are said to be
linear. Datasets that admit such separation are called **linearly
separable**.

While in regression problems the target values were continuous, in
classification they need to represent the K discrete class labels. There
are various ways of doing that. For instance, if $K=2$, we have a binary
problem in which we can simply take $$y \in \{0,1\}$$, corresponding to
**negative classes** or **positive classes**, respectively. For $K > 2$,
namely multiclass problems, it is convenient to introduce the
**$\pmb{1}$-of-K coding** scheme in which we first define a discrete
output variable $$y \in \{1, 2, \dots, K\}$$ and then map this function to
a vector $\pmb{T}(y) \in \mathbb{R}^K$ for all $y$ such that

$$\label{1ofKcoding}
	\pmb{T}(1) = \begin{pmatrix}
	1\\
	0\\
	0\\
	\vdots\\
	0
	\end{pmatrix}\,, \quad  \pmb{T}(2) = \begin{pmatrix}
	0\\
	1\\
	0\\
	\vdots\\
	0
	\end{pmatrix}\,, \quad \dots \quad \pmb{T}(K) = \begin{pmatrix}
	0\\
	0\\
	0\\
	\vdots\\
	1
	\end{pmatrix}\,.$$
  
Obviously, this same scheme can be applied to the
binary case, but this is not common in practice. There is also a variant
of 1-of-K coding where we take $\pmb{T}(y) \in \mathbb{R}^{K-1}$
instead, but one class is represented as the zero vector. Such a
convention is more similar to the one used for binary classification.

One approach to define the decision boundaries, and with them assign
each input to a given class, is through the use of non-linear functions
that map inputs $\pmb{x}\in \mathbb{R}^D$ to a $(0, 1)$ range:

$$g_k(\pmb{\Theta}_k^T \pmb{x}) \in (0, 1)\,, \quad k=1, \dots, K\,.$$

Here $\pmb{\Theta}\in \mathbb{R}^{D}$ are the model's parameters and the
$g_k(\, \cdot\, )$ are usually called **activation functions** and their
inverses **link functions**. Restricting the output to lie between $0$
and $1$ allows us to interpret it as a probability. For it, however, we
need the extra constraint

$$\sum_{k=1}^{K} g_k(\pmb{\Theta}_k^T \pmb{x}) = 1 \quad \forall \pmb{x}\,,$$

which means that each input must belong to at least one class. The
assigned class to each input $\pmb{x}$ is then easily read via

$$y = \underset{k \in \{1, \dots, K\}}{argmax} \ g_k(\pmb{\Theta}_k^T \pmb{x})\,,$$

corresponding to the maximum probability. With this label we then build
$\pmb{T}(y)$ trough the 1-of-K coding scheme mentioned above.

The decision boundaries can also be obtained from the $g_k$. Indeed, the
surface separating two classes is where the probabilities of belonging
to those two classes are equal. For instance, the boundary between the
classes $C_1$ and $C_2$ is defined by the condition

$$g_1(\pmb{\Theta}_k^T \pmb{x}) = g_2(\pmb{\Theta}_k^T \pmb{x})\,.$$ 

In particular, if this condition leads to a linear constraint on $\pmb{x}$,
we get a linear model. This could happen, for instance, if all
activation functions are equal $g_k(\, \cdot\, ) \equiv g(\, \cdot\, )$
for all $k$. Then,

$$g(\pmb{\Theta}_k^T \pmb{x}) = g(\pmb{\Theta}_j^T \pmb{x}) \quad \Leftrightarrow \quad (\pmb{\Theta}_k - \pmb{\Theta}_j)^T\pmb{x}= 0\,,$$

which is clearly a linear problem, as the constraint is nothing but the
equation for a hyperplane with normal $\pmb{\Theta}_k - \pmb{\Theta}_j$.
This is exactly the case for logistic regression and softmax regression.

## Logistic regression

We now introduce logistic regression, a linear model for binary
classification. Like in linear regression, our first step is to propose
a hypothesis function which maps inputs into predicted outputs. The
latter are taken to be one of two classes, $0$ or $1$, and so we need an
activation function whose image lies in that range. A good candidate for
such a job is 

\begin{equation}\label{eq: h}
	h_{\pmb{\Theta}}(\pmb{x}) = g(\pmb{\Theta}^T \pmb{x}) = \frac{1}{1+e^{-\pmb{\Theta}^T \pmb{x}}}\,,
\end{equation} 

where we choose the so-called **logistic function** or **sigmoid function**

\begin{equation}\label{eq: sigmoid}
	g(z) \equiv \frac{1}{1 + e^{-z}}\,,
\end{equation}
  
and we already included the intercept term in
$\pmb{\Theta}^T \pmb{x}= \Theta_0 + \Theta_1 x_1 + \dots + \Theta_D x_D$.
The sigmoid function satisfies $\lim_{z \to - \infty}g(z) = 0$ and
$\lim_{z \to \infty} g(z) = 1$ so it is bounded between $0$ and $1$. It
also obeys the following useful identity under derivation:

\begin{equation}\label{eq: g'}
	g'(z) = g(z)(1 - g(z))\,.
\end{equation}

The decision boundary in this case is determined by

$$g(\pmb{\Theta}^T \pmb{x}) = \frac12 \Leftrightarrow  \pmb{\Theta}^T \pmb{x}= 0\,,$$

defining a linear problem.

Given the model and an $N$-dimensional training set
$(\pmb{x}^{(i)},y^{(i)})_{i=1}^{N}$, we would like to apply an
optimization algorithm to train the parameters, $\pmb{\Theta}$, so we
get the best fit for such examples. Remembering what we did for linear
regression, the first step would be to define a cost function to
minimize. However, we learned that the very same result could be
obtained by some probabilistic assumptions. Here we take directly the
latter approach as our starting point by defining the log likelihood and
applying the maximum likelihood principle. To this end, while for linear
regression the examples followed a normal distribution, here in logistic
regression we assume the probability distribution for each example to be

$$\label{bernu}
	P(y=1 | \pmb{x}; \pmb{\Theta}) \equiv h_{\pmb{\Theta}}(\pmb{x})\,, \quad P(y=0 | \pmb{x}; \pmb{\Theta}) \equiv 1 - h_{\pmb{\Theta}}(\pmb{x})\,,$$

which can be written in a compact form as

$$P(y | \pmb{x}; \pmb{\Theta}) \equiv \left[h_{\pmb{\Theta}}(\pmb{x})\right]^y \left[1-h_{\pmb{\Theta}}(\pmb{x})\right]^{1-y}\,.$$

This corresponds to a **Bernoulli distribution**.

Assuming that the $N$ examples were identically and independently
distributed (i.i.d), the likelihood is given by

$$L(\pmb{\Theta}) \equiv \prod_{i=1}^{N} \left[h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right]^{y^{(i)}} \left[1-h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right]^{1-y^{(i)}}\,,$$

and the log likelihood (which is easier to maximize) takes the form

$$\label{eq: l}
	\ell(\pmb{\Theta}) \equiv \sum_{i=1}^{N} \left\{y^{(i)} \log \left[h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right] + (1-y^{(i)})\log \left[1-h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right]\right\}\,.$$

Now that we have $\ell(\pmb{\Theta})$, we can maximize it by any of the
methods we learned previously, namely, gradient descent (GD) or its
variants, stochastic gradient descent (SGD) and mini-batch SGD. In all
those cases, we need the gradient of $\ell(\pmb{\Theta})$, which for a
single example would look like:

$$\nabla_{\pmb{\Theta}}\ell(\pmb{\Theta}) = \left(y - h_{\pmb{\Theta}}(\pmb{x})\right) \pmb{x}\,,$$

where we used the identity for the logistic function $\eqref{eq: g'}$. So, for instance, for SGD, in which we use a single example $(\pmb{x}^{(i)}, y^{(i)})$ at the time, the update rule reads

$$\label{eq: thetaSGD}
	\pmb{\Theta}:= \pmb{\Theta}+ \alpha \left(y^{(i)} - h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right) \pmb{x}^{(i)}\,,$$

where we introduced a learning rate $\alpha$. Technically speaking, $\eqref{eq: thetaSGD}$ is called **stochastic gradient ascent** since we *add* the gradient in each step, instead of subtracting it, this is
because we want to *maximize* $\ell(\pmb{\Theta})$. As a side remark, it
is worth noticing that $\eqref{eq: thetaSGD}$ takes exactly the same form as for linear
regression yet now with $h_{\pmb{\Theta}}(\pmb{x})$ given by
$\eqref{eq: h}$ instead of $h_{\pmb{\Theta}}(\pmb{x}) = \pmb{\Theta}^T \pmb{x}$. This very same
formal structure is not an accident and we will understand its origin
when analyzing general linear models in following notes.

### A diversion: the Newton-Raphson method

Apart from GD and its variants, there are other optimization methods one
could propose, in this subsection we will study one of them called
**Newton's method**. It is worth emphasizing that this optimizer is by
no means restricted to logistic regression problems, and we could have
introduced it earlier with linear regression, for instance.

Newton's method can be viewed as an algorithm to find the root of a
given function. For instance, suppose we have
$f: \mathbb{R}\mapsto \mathbb{R}$ and we want to find
$\hat \Theta\in \mathbb{R}$ such that $f(\hat \Theta) = 0$, then
Newton's update rule reads 

\begin{equation}\label{eq: Newton_rule}
	\Theta:= \Theta - \frac{f(\Theta)}{f'(\Theta)}\,,
\end{equation}
  
where the prime denotes derivative with respect to the argument. This method can be
applied to the problem of maximizing a 1-dimensional log likelihood
since the extremum point is achieved when $\ell'(\Theta) = 0$. In this
case the update rule reads 

\begin{equation}\label{eq: Newton_rule_ell}
	\Theta:= \Theta - \frac{\ell'(\Theta)}{\ell''(\Theta)}\,,
\end{equation}
  
which is exactly $\eqref{eq: Newton_rule}$ with $f(\Theta) = \ell'(\Theta)$.

Since in our problem $\pmb{\Theta}$ is vector-valued, then we need the
higher-dimensional generalization of $\eqref{eq: Newton_rule_ell}$ which takes the name of **Newton-Raphson method** and is given by

$$\pmb{\Theta}:= \pmb{\Theta}- \alpha \pmb{H}^{-1} \nabla_{\pmb{\Theta}}\ell(\pmb{\Theta})\,,$$

where we introduced a learning rate $\alpha$ and the inverse of the
**Hessian matrix**

$$H_{i j} \equiv \frac{\partial^2 \ell(\pmb{\Theta})}{\partial \Theta^i \partial \Theta^j}\,.$$

Newton's method typically is faster than GD and its variants and
requires less iterations to converge. However, each iteration can be
more expensive because we compute and invert a $(D+1)\times(D+1)$ matrix
for each step, which for large data sets could take a lot of time. As a
side remark, when Newton-Raphson is applied to maximize the log
likelihood function for logistic regression, the method is also called
**Fisher scoring**.

### The Perceptron algorithm

Before moving to the multi-class problem, is worth mentioning an
algorithm which will come back later and can be seen as a more primitive
version of logistic regression. This model is given by considering a
modification of the logistic regression problem where we force the
hypothesis function to be exactly either $0$ or $1$, i.e.

$$h_{\pmb{\Theta}}(\pmb{x}) = g(\pmb{\Theta}^T \pmb{x}) \equiv \begin{cases}
	1 &\text{if} \ \pmb{\Theta}^T \pmb{x}\geq 0\\
	0 &\text{if} \ \pmb{\Theta}^T \pmb{x}< 0
	\end{cases}\,.$$

With this model, the update rule is analogous to $\eqref{eq: thetaSGD}$ but now with this new step function

$$\pmb{\Theta}:= \pmb{\Theta}+ \alpha \left(y^{(i)} - h_{\pmb{\Theta}}(\pmb{x}^{(i)})\right) \pmb{x}^{(i)}\,,$$

which defines the **perceptron learning algorithm**. Even though the
update rule is formally the same as for linear and logistic regression,
the perceptron algorithm is not deduced from probabilistic assumptions
or by imposing a maximum likelihood principle.

## Softmax regression

We now turn to a multi-class problem where each $D$-dimensional input
$\pmb{x}$ can belong to one of $K$ classes $C_k$. We encode this
information in vector outputs using the 1-of-$K$ coding scheme
introduced above. To this end, we first label each class by
$$y \in \{1, 2, \dots, K\}$$ and then map them to
$\pmb{T}(y) \in \mathbb{R}^K$ using
1-of-K coding. We introduce $K$ parameter vectors, namely
$\pmb{\Theta}_i$ with $i=1, \dots, K$ and with them we define $K$
hypothesis functions

$$h_{\pmb{\Theta}}(\pmb{x})_i = g(\pmb{\Theta}_i^T \pmb{x}) = \frac{e^{\pmb{\Theta}_i^T \pmb{x}}}{\sum_{j=1}^{K} e^{\pmb{\Theta}_j^T \pmb{x}}}\,.$$

All of them are defined in terms of the same function, the softmax
function, and the only difference for each class enters through the
parameters. This is the softmax regression model and it is built such
that 

$$\label{softmaxprop}
	h_{\pmb{\Theta}}(x)_i \in (0, 1) \ i=1, \dots, K\,, \quad \sum_{i=1}^{K} h_{\pmb{\Theta}}(x)_i = 1\,.$$

It is interesting to notice that the normalization condition implies
that one of the vector parameters $\pmb{\Theta}_i$ can be determined
from the rest. As a consequence, the model is truly characterized by
$D \times (K-1)$ parameters, not $D \times K$.

Under derivation, the softmax function satisfies a similar identity to
the one obeyed by the logistic function $\eqref{eq: g'}$:

$$\nabla_{\pmb{\Theta}_i} g(\pmb{\Theta}_j^T \pmb{x}) = g(\pmb{\Theta}_i^T \pmb{x}) \left[\delta_{i j} - g(\pmb{\Theta}_j^T \pmb{x})\right] \pmb{x}\,,$$

where $\delta_{i j}$ is the Kronecker delta and we are not assuming
summation on repeated indices. The decision boundary between each pair
of classes, let's say $i$ and $j$, is defined via the constraint

$$(\pmb{\Theta}_i - \pmb{\Theta}_j)^T \pmb{x}= 0\,.$$

In order to train the model, we need the log likelihood and therefore we
need the conditional probability distributions
$P(y=i| \pmb{x}; \pmb{\Theta})$ for all classes. The Bernoulli
assumption for logistic regression admits a generalization for the
multiclass case in which we model the problem with a **multinomial
distribution**: 

$$\label{multinomial}
	P(y = i | \pmb{x}; \pmb{\Theta}) = h_{\pmb{\Theta}_i}(\pmb{x})\,, \quad \sum_{i=1}^{K}P(y = i | \pmb{x}; \pmb{\Theta}) = 1\,,$$

where the latter identity is derived from
\eqref{softmaxprop}. For $K=2$ and after a change of conventiones, \eqref{multinomial} reduces to \eqref{bernu}.

As for logistic regression, here we can also write the conditional
probability distribution for all classes in a compact form:

$$P(y | \pmb{x}; \pmb{\Theta}) = \prod_{i=1}^{K} h_{\pmb{\Theta}_i}(\pmb{x})^{1\{y=i\}}\,,$$

where we introduced the **indicator function** $1{\cdot}$ defined by

$$1\{\text{True}\} = 1, \quad 1\{\text{False}\} = 0\,.$$ 

For instance, $$1 \{3=2\} = 0$$ and $$1\{\text{Maradona} > \text{Pelé}\} = 1$$. For our
particular case, we could also have written the exponent in terms of the
Kronecker delta $$1\{y=i\} = \delta_{i y}$$.

Using this probability distribution and assuming the $N$-dimensional
dataset is i.i.d. the log likelihood is given by

$$\ell(\pmb{\Theta}) = \sum_{i=1}^{N} \log P(y^{(i)} | \pmb{x}^{(i)}; \pmb{\Theta}) =\sum_{i=1}^{N} \sum_{j=1}^{K} 1\{y^{(i)} = j\} \log \frac{e^{\pmb{\Theta}_j^T \pmb{x}^{(i)}}}{\sum_{l=1}^{K}e^{\pmb{\Theta}_l^T \pmb{x}^{(i)}}}\,,$$

where we used the definition of the multinomial distribution in the
product form together with the explicit expression for
$h_{\pmb{\Theta}_i}(\pmb{x})$. With $\ell(\pmb{\Theta})$ we can fit the
parameters by imposing the maximum likelihood principle together with
our favorite optimizer.