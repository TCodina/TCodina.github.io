---
title: 'Generalized Linear Models'
date: 2023-09-20
permalink: /notes/generalized_linear_models
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
\newcommand{\bg}{\pmb{g}}
$

In previous notes we have seen how certain models emerged naturally from
probabilistic assumptions on the dataset: linear regression is defined
by assuming a Gaussian conditional distribution between targets and
features, logistic regression requires a Bernoulli distribution, and the
softmax regression model uses a multinomial distribution. In these notes
we study **generalized linear models (GLMs)**, a larger class of models
containing the previous ones, and many others, as particular cases. We
begin introducing GLMs, and then analyze how the simpler models we
studied before enter in this bigger picture. We finalize by studying the
**Poisson regression model**, as a particular case of GLMs.

Good references for an introduction to GLMs are:

-   Lecture 6 of **Stanford's CS229 course of 2019**:
    [video-lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh),
    [pdf and slides](http://cs229.stanford.edu/syllabus-summer2019.html)
    and [problem sets](http://cs229.stanford.edu/summer2019/)

-   [These lecture
    notes](https://www.cs.princeton.edu/~bee/courses/scribe/lec_09_18_2013.pdf)
    from **Princeton's STA561c course of 2013**.

## The exponential family

We begin defining the **exponential family**: 

$$\label{eq: GLM}
    P(\pmb{y}; \pmb{\eta}) = b(\pmb{y}) e^{\left[\pmb{\eta}^T \pmb{T}(\pmb{y}) - a(\pmb{\eta})\right]}\,,$$

a parametric set of probability distributions. Here $\pmb{\eta}$ is
called the **natural or canonical parameter** of the distribution,
$\pmb{T}(\pmb{y})$ is the **sufficient statistic**, $b(\pmb{y})$ is the
base measure and $a(\pmb{\eta})$ is the **log partition function**
which, inside $e^{- a(\pmb{\eta})}$, just plays the role of a
normalization constant so the sum/integral adds up to $1$ 

$$\label{a}
    \int d\pmb{y}\, P(\pmb{y}; \pmb{\eta}) = e^{-a (\pmb{\eta})} \int d\pmb{y}\, b(\pmb{y}) e^{\pmb{\eta}^T \pmb{T}(\pmb{y})} = 1 \quad \Rightarrow \quad a(\pmb{\eta}) = \log \left[\int d\pmb{y}\, b(\pmb{y}) e^{\pmb{\eta}^T \pmb{T}(\pmb{y})}\right]\,.$$

The latter identity justifies the name for $a$. Taking the gradient
w.r.t. $\pmb{\eta}$ on both sides, we arrive at a powerful relation that
will become useful later on: 

$$\label{nablaa}
    \nabla_{\pmb{\eta}} a(\pmb{\eta}) = \int d\pmb{y}\, \pmb{T}(\pmb{y}) b(\pmb{y}) e^{\left[\pmb{\eta}^T \pmb{T}(\pmb{y}) - a(\pmb{\eta})\right]} = E[\pmb{T}(\pmb{y}); \pmb{\eta}]\,.$$

Here we used the exact expression for $a(\pmb{\eta})$ coming from
\eqref{a} and recognized the expectation value w.r.t. the exponential family \eqref{eq: GLM}.

A fixed choice for $b,a$ and $\pmb{T}$ defines a family of distributions
parameterized by $\pmb{\eta}$. Each element of the family is reached by
a different value of $\pmb{\eta}$. We now recover other more basic
distributions as particular cases of the exponential family.

### Gaussian

The Gaussian distribution is a member of the exponential family. To see
this, we need to remember the definition of the normal distribution and
rewrite it in a way that can be easily compared with
\eqref{eq: GLM}. For the simpler case in which $\sigma^2=1$, we have

$$P(y; \mu) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}(y-\mu)^2} = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}y^2} e^{\mu y - \frac12 \mu^2}\,,$$

which is written in the form of \eqref{eq: GLM} with

$$
    \eta = \mu\,,\\
    T(y) = y\,,\\
    a(\eta) = \frac12 \mu^2 = \frac12 \eta^2\,,\\
    b(y) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}y^2}\,.\label{GaussianasGLM}$$

### Bernoulli

In the same spirit, one can probe that the Bernoulli distribution also
belongs to the exponential family. To see this, we remember that
Bernoulli is a distribution over $$y\in\{0,1\}$$ with mean $\phi$. In this
context, different values of $\phi$ specifies different elements of the
family. We can rewrite the distribution as follows: 

$$P(y; \phi) = \phi^y \phi^{1-y} = e^{\log \left(\frac{\phi}{1-\phi}\right) y + \log(1-\phi)}\,.$$ 
    
From this rewriting we can recognize:

$$\label{bernoulliasGLM}
    
    \eta = \log \left(\frac{\phi}{1-\phi}\right) \quad \Rightarrow \quad \phi = \frac{1}{1 + e^{-\eta}}\,,\\
    T(y) = y\,,\\
    a(\eta) = -\log(1-\phi) = \log(1+e^{\eta})\,,\\
    b(y) = 1\,,
    $$ 
    
thus probing that the Bernoulli distribution is a
particular case of the exponential family. As a side remark, it is worth
noticing that the mean value $\phi$ depends on the canonical parameters
through our previously introduced sigmoid function.

### Multinomial

Finally, we show that the multinomial distribution is also a member of
the exponential family. Such distribution is characterized by $K$
parameters $\phi_1, \dots, \phi_K$ which determine the probability of
each outcome $$y \in \{1, \dots, K\}$$. It can be written in a compact
form as 

$$P(y; \phi) = \prod_{i=1}^{K}\phi_i^{1\{y=i\}}\,,$$ 

where we used the indicator function $$1\{\cdot\}$$ defined by
$$1\{\text{True}\} = 1$$ and $$1\{\text{False}\} = 0$$. Since the sum of all
the probabilities must return $1$, we really have $K-1$ parameters
together with $\phi_K = 1 - \sum_{i=1}^{K-1} \phi_i$. However, we will
keep $\phi_K$ just for convenience.

Unlike the previous examples, here $\pmb{T}(y) \neq y$. Instead, we
remember the 1-of-K coding scheme and introduce a $(K-1)$-dimensional
vector function, $\pmb{T}(y) \in \mathbb{R}^{K-1}$, taking the following
values: 

$$T(1) = \begin{pmatrix}
    1\\
    0\\
    0\\
    \vdots\\
    0
    \end{pmatrix}\,, \quad T(2) = \begin{pmatrix}
    0\\
    1\\
    0\\
    \vdots\\
    0
    \end{pmatrix}\,, \quad \dots \quad T(K-1) = \begin{pmatrix}
    0\\
    0\\
    0\\
    \vdots\\
    1
    \end{pmatrix}\,, \quad T(K) = \begin{pmatrix}
    0\\
    0\\
    0\\
    \vdots\\
    0
    \end{pmatrix}\,.$$

Denoting the $i$th component of the vector as $T(y)_i$ (such that, for
instance, $T(1)_1 = 1$ and $T(2)_1 = 0$) the rewriting of the
multinomial distribution in the form of \eqref{eq: GLM} goes as follows: 

$$\label{eq: multinomial_as_exponential}
    
    P(y; \phi) = \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \dots \phi_K^{1\{y=K\}}\\
    = \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \dots \phi_K^{1 - \sum_{i=1}^{K-1} 1\{y=i\}}\\
    = \phi_1^{T(y)_1} \phi_2^{T(y)_2} \dots \phi_K^{1 - \sum_{i=1}^{K-1} T(y)_i}\\
    = e^{\left[  T(y)_1\log\left(\frac{\phi_1}{\phi_K}\right) + T(y)_2\log\left(\frac{\phi_2}{\phi_K}\right) + \dots + T(y)_{K-1}\log\left(\frac{\phi_{K-1}}{\phi_K}\right) + \log(\phi_K) \right]}\\
    = b(y) e^{\left[ \pmb{\eta}^T \pmb{T}(y) - a(\pmb{\eta})\right]}\,,
    $$ 
    
with the elements 

$$\label{softmaxGLM}

\eta_i= \log \left(\frac{\phi_i}{\phi_K}\right)\,,\quad i=1, \dots, K-1\,,\\
T(y)_i = 1\{y=i\}\,,\quad i=1, \dots, K-1\,,\\
a(\pmb{\eta}) = - \log(\phi_K)\,,\\
b(y) = 1\,.
$$

These three distributions are just some examples of exponential families
but there are many more, e.g. Poisson, gamma, exponential, beta,
Dirichlet, just to name a few. To see how all these distributions are
related to exponential families, there is a very nice table in this
[Wikipedia page](https://en.wikipedia.org/wiki/Exponential_family).

## Constructing GLMs

So far we have seen how starting from a dataset that is assumed to
follow certain distribution we can build learning models. To this end
one introduces learnable parameters $\pmb{\Theta}$ and uses the
**maximum likelihood estimate (MLE)** to find their optimal values. We
followed this path using a Gaussian distribution to arrive at the least
mean square algorithm, Bernoulli to obtain logistic regression and
multinomial for softmax regression. In the same spirit, we can assume
our dataset follows a distribution of the exponential family and derive
learning models called **Generalized linear models (GLMs)**.

As usual, we begin with a dataset made of $D$-dimensional features
$\pmb{x}$ and targets $\pmb{y}$. Our task is then to learn the
functional relationship between these two. To this end, GLMs are built
from the following two assumptions or \"design choices\":

1.  We assume that $P(\pmb{y}| \pmb{x}; \pmb{\Theta})$ is given by an
    exponential family distribution where $\pmb{\eta}$ and the inputs
    $\pmb{x}$ follow a linear relation: 
    
    $$\label{asu1}
            \eta_i = \pmb{\Theta}^T_i \pmb{x}\,.$$

2.  Given $\pmb{x}$, the hypothesis function is given by the expected
    value of $\pmb{T}(\pmb{y})$, namely 
    
    $$\label{asu2}
            h_{\pmb{\Theta}}(\pmb{x})_i = E[T(\pmb{y})_i;\pmb{\eta}] = g(\pmb{\eta})_i\,,$$
    
    in terms of the so-called **canonical response function**,
    $\pmb{g}(\pmb{\eta})$ whose inverse, $\pmb{g}^{-1}(\pmb{\eta})$, is
    called **canonical link function**.

Here each $\pmb{\Theta}_i$ is a $D$-dimensional parameter vector and
$i = 1, \dots, \text{dim}(\pmb{\eta})$. Equation \eqref{nablaa} gives us an alternative way of computing the hypothesis function simply as

$$\label{trick}
    \pmb{h}_{\pmb{\Theta}}(\pmb{x}) = \nabla_{\pmb{\eta}} a(\pmb{\eta})\,.$$

We now explore how several learning algorithms are particular cases of
GLMs under these assumptions.

### Linear regression

In the case of linear regression we model the conditional distribution
of $y$ given $\pmb{x}$ as a Gaussian. The hypothesis function is then
given by: 

$$
        h_{\pmb{\Theta}}(\pmb{x}) = E[T(y)|\pmb{x}; \pmb{\Theta}] = E[y|\pmb{x}; \pmb{\Theta}] = \mu = \eta = \pmb{\Theta}^T \pmb{x}\,.
$$

The first equality follows from assumption 2. while in the second
equality we used \eqref{GaussianasGLM}. The third equality is the standard result
for Gaussian distributions, the fourth equality comes from
\eqref{GaussianasGLM}, and the last equality follows from
assumption 1. The final result is the hypothesis function for linear
regression! Note that the same result holds if one uses
\eqref{trick} with
$a(\eta) = \tfrac12 \eta^2$ and $\eta = \pmb{\Theta}^T \pmb{x}$. In the
language of GLMs, from \eqref{asu2} we recognize the canonical response function as the
identity function.

### Logistic regression

Now, the response variable is binary-valued so we model the problem with
a Bernoulli distribution and we follow similar steps as before to obtain
the hypothesis function 

$$
        h_{\pmb{\Theta}}(\pmb{x}) = E[T(y)|\pmb{x}; \pmb{\Theta}] = E[y|\pmb{x}; \pmb{\Theta}] = \phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\pmb{\Theta}^T \pmb{x}}}\,.
$$

Here, the sigmoid function emerges as the response function of the
corresponding GLM.

### Softmax regression

Following similar steps as for linear and logistic regression, the
hypothesis function for this multiclass problem reads 

$$\label{hsoftmax}
h_{\pmb{\Theta}}(\pmb{x})_i = E[T(y)_i|\pmb{x}; \pmb{\Theta}]_i = P(y=i|\pmb{x};\pmb{\Theta}) = \phi_i(\pmb{\eta})\,,$$

where $\phi_i(\pmb{\eta})$, playing the role of canonical response
function, is obtained by inverting $\eta_i(\pmb{\phi})$ from
\eqref{softmaxGLM}. To this end we first extend the definition of
$\eta_i$ to include also
$\eta_K = \log \left(\frac{\phi_K}{\phi_K}\right) = 0$ and then invert
the log 

$$\label{eq: auxiliar1}
    \eta_i= \log \left(\frac{\phi_i}{\phi_K}\right) \quad \Rightarrow \quad \phi_K e^{\eta_i} = \phi_i\,.$$

Now we sum over all $K$ components and use the normalization condition to get

$$\phi_K \sum_{i=1}^{K}e^{\eta_i} = \sum_{i=1}^{K} \phi_i = 1 \quad \Rightarrow \quad \phi_K = \frac{1}{\sum_{i=1}^{K}e^{\eta_i}}\,.$$

Plugging back this result into
\eqref{eq: auxiliar1} we obtain the response function

$$\label{phietasoftmax}
\phi_i(\pmb{\eta}) = \frac{e^{\eta_i}}{\sum_{j=1}^{K}e^{\eta_j}}\,,$$

which is our old friend, the softmax function. By plugging
\eqref{phietasoftmax} into \eqref{hsoftmax}, together with the linear assumption \eqref{asu1}, we get the
expected hypothesis function for softmax regression:

$$\label{eq: softmax_regression}
h_{\pmb{\Theta}}(\pmb{x})_i = \frac{e^{\pmb{\Theta}_i^T \pmb{x}}}{\sum_{j=1}^{K} e^{\pmb{\Theta}_j^T \pmb{x}}}\,.$$

### Maximum likelihood estimate

These three examples helped us to understand that the hypothesis
functions we were proposing for the simple models are just a consequence
of describing the problem with a bigger class of learning models. On the
same lines, one can see that the MLE of $\pmb{\Theta}$ for all these
models have the same structure! To see this, we introduce an
$N$-dimensional training set
$$\{ \pmb{x}^{(i)}, \pmb{y}^{(i)} \}_{i=1}^{N}$$ and build the log
likelihood for \eqref{eq: GLM} with $\pmb{\eta} = \pmb{\Theta}^T \pmb{x}$

$$\ell(\pmb{\Theta}) = \sum_{i=1}^{N} \log P(\pmb{y}^{(i)}| \pmb{x}^{(i)}; \pmb{\Theta}) = \log b (\pmb{y}^{(i)}) + (\pmb{x}^{(i)})^T\, \pmb{\Theta}\, \pmb{T}(\pmb{y}^{(i)}) - a(\pmb{\Theta}^T \pmb{x}^{(i)})\,.$$

We can now take the gradient w.r.t. $\pmb{\Theta}$ to obtain

$$\nabla_{\pmb{\Theta}} \ell(\pmb{\Theta}) = \sum_{i=1}^{N} \left[\pmb{T}(\pmb{y}^{(i)}) - \pmb{h}_{\pmb{\Theta}}(\pmb{x}^{(i)})\right] \pmb{x}^{(i)}\,,$$

where we used \eqref{trick}. With this gradient we can then build the
corresponding learning rule. For GD, for instance, we have 

$$\label{GD}
\pmb{\Theta}:= \pmb{\Theta}+ \alpha \sum_{i=1}^{N} \left[\pmb{T}(\pmb{y}^{(i)}) - \pmb{h}_{\pmb{\Theta}}(\pmb{x}^{(i)})\right] \pmb{x}^{(i)}\,,$$

with $\alpha$ the learning rate. It is easy to check that
\eqref{GD} is indeed the same rule we derived for linear, logistic and softmax regression when
introducing the corresponding sufficient statistic and hypothesis
function!

We now have a powerful recipe to build and train a huge class of
learning models on a given dataset that can be summarized in the
following steps:

1.  Propose a member of the exponential family distribution to describe
    the dataset and identify the corresponding $b, a$ and $\pmb{T}$
    
    $$P(\pmb{y}| \pmb{x}; \pmb{\Theta}) = b(\pmb{y}) e^{\left[\pmb{\eta}^T \pmb{T}(\pmb{y}) - a(\pmb{\eta})\right]}\,, \quad \text{with} \quad \pmb{\eta} = \pmb{\Theta}^T \pmb{x}\,.$$

2.  The hypothesis function is then given by 

$$\label{step2}
        \pmb{h}_{\pmb{\Theta}}(\pmb{x}) = \left.\nabla_{\pmb{\eta}} a(\pmb{\eta})\right|_{\pmb{\eta} = \pmb{\Theta}^T \pmb{x}}\,.$$

3.  Fit the parameters by using the MLE with 

$$\label{step3}
        \nabla_{\pmb{\Theta}} \ell(\pmb{\Theta}) = \sum_{i=1}^{N} \left[\pmb{T}(\pmb{y}^{(i)}) - \pmb{h}_{\pmb{\Theta}}(\pmb{x}^{(i)})\right] \pmb{x}^{(i)}\,,$$
    
together with your favorite optimization rule.

In the following section we use this recipe to build a new model, the
Poisson regression model.

## Poisson regression model

Recovering all results from more general ones is always fun, but it is
definitely more fun to study the new predictions coming from the
generalization. Because of this, here we present the Poisson regression
model following the recipe presented above.

A Poisson distribution is a probability distribution that describes the
number of rare, discrete events $$y \in \{1, 2, \dots\}$$ occurring in a
fixed interval of time or space when these events happen randomly and
independently at a constant average rate. It is characterized by a
single parameter, $\lambda$, which represents the average rate of
occurrence 

$$\label{Poisson}
P(y; \lambda) = \frac{\lambda^y}{y!} e^{-\lambda}\,.$$

It is a member of the exponential family, characterized by the following
data: 

$$
\eta = \log \lambda \quad \Rightarrow \quad \lambda = e^{\eta}\,,\\
T(y) = y\,,\\
a(\eta) = \lambda = e^{\eta}\,,\\
b(y) = \frac{1}{y!}\,,
$$ 

which can be easily derived by rewriting \eqref{Poisson} in the form of \eqref{eq: GLM}.

We now build a learning model from a i.i.d dataset
$$\{\pmb{x}^{(i)}, y^{(i)} \}_{i=1}^{N}$$ following such distribution.
Introducing a $D$-dimensional vector of learnable parameters
$\pmb{\Theta}$ we use \eqref{step2} to derive the corresponding hypothesis function

$$\label{Poissonh}
h_{\pmb{\Theta}}(\pmb{x}) = e^{\pmb{\Theta}^T \pmb{x}}\,,$$ 

which defines the **Poisson regression model**. Finally, if we want to train this
model, we can simply use \eqref{step3}

$$\nabla_{\pmb{\Theta}} \ell(\pmb{\Theta}) = \sum_{i=1}^{N} \left[y^{(i)} - e^{\pmb{\Theta}^T \pmb{x}^{(i)}} \right] \pmb{x}^{(i)}\,,$$

for a MLE of $\pmb{\Theta}$.