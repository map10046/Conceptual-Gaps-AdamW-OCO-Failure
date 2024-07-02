# Conceptual Gaps AdamW OCO Failure
 A large portion of my project for conceptual gaps about online convex optimization and how AdamW can fail with reasonable parameters on a simple problem. For a summary, check out the [poster](Poster/gaps_poster.pdf). 

 # Initial Directions 

 I originally was interested in the correlation of Adam/SGD weights because I figured if I could see paths in an epoch $\times$ epoch matrix of $[0,1]-$valued correlations, it could give insight to when they follow the same path toward a local minima, or potential divergence criterion. An example matrix is at the end of the [CIFAR notebook](Notebooks/cifar-training.ipynb). 

 I was also interested in divergences where simple functions gave issues for Adam-type optimizers but not SGD. Following some ideas mentioned in Reddit et. al. (On the convergence of Adam and Beyond), they remark large differences in the gradient could pose issues for Adam because it uses global $\beta$ parameters and those should be tailored based on gradient behavior. I show bad $\beta$ values can give divergence of Adam on simple quadratics in this [1D notebook](Notebooks/Optim_Paths.ipynb). I was able to slow convergence of Adam-type optimizers in this [2D notebook](Notebooks/Paths-3d.ipynb) using quadratics like $f(x,y) = \alpha x^2 + \beta y^2$ with large differences in $\alpha,\beta$.

 # Online Convex Optimization (OCO) 

 In Reddi et. al. (On the Convergence of Adam and Beyond), the authors show Adam can fail on a simple (linear) online convex optimization problem with $\beta$ values which are allowed in the theorem of the original Adam paper. I break down their proof [here](Poster/proof.pdf) and I adapt it to show AdamW also fails on this problem as long as the weight decay is kept to a reasonable value. I show their example and the failure of AdamW in [this notebook](Notebooks/online_counterexamples.ipynb). 
