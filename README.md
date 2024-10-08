# Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation

# Reference:
1, Shao, S., Zhou, Z., Chen, H., & Shen, Z. (2024). Elucidating the Design Space of Dataset Condensation. arXiv preprint arXiv:2404.13733.

# Elucidate Dataset Condensation (EDC)

- The prohibitively expensive bi-level optimization paradigm limits the effectiveness of traditional dataset distillation methods
- While uni-level optimization has demonstrated remarkable performance on large datasets, it has yet to achieve the benchmark accuracy levels seen with classical methods on datasets like CIFAR-10/100
- The newly developed training-free RDED significantly outperforms previous methods in efficiency and maintains effectiveness, yet it overlooks the potential information loss due to the lack of image optimization
- Some simple but promising techniques (e.g., smoothing the learning rate schedule) that could enhance performance have been underexplored in existing literature.

These drawbacks show the constraints of previous methods in several respects, highlighting the need for a thorough investigation and assessment of these issues. Therefore, Elucidate Dataset Condensation (EDC), which includes a range of detailed and effective enhancements, is developed to remedy these drawbacks of dataset distillation. 

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/edc.png)

# Dataset Condensation

Dataset condensation, also known as dataset distillation, has emerged in response to the ever-increasing training demands of advanced deep-learning models. This approach addresses the challenge of requiring high-precision models while also managing substantial resource constraints. In this method, the original dataset acts as a “teacher”, distilling and preserving essential information into a smaller, surrogate “student” dataset. The ultimate goal of this technique is to achieve performance comparable to the original by training models from scratch with the condensed dataset. This approach has become popular in various downstream applications, including continual learning, neural architecture search, and training-free network slimming.

Dataset condensation involves generating a synthetic dataset:

$D^S := \{(x_{i}^{S}, y_{i}^{S})\}_{i=1}^{|D^S|}$ 

- consisting of images $X^S$ and labels $Y^S$

Designed to be as informative as the original dataset:

$D^T := \{(x_{i}^{T}, y_{i}^{T})\}_{i=1}^{|D^T|}$ 

- includes images $X^T$ and labels $Y^T$

The synthetic dataset $D^S$ is substantially smaller in size than $D^T$($|D^S|$ $≪$ $|D^T|$). The goal of this process is to maintain the
critical attributes of $D^T$ to ensure robust or comparable performance during evaluations on test protocol $P_D$.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/dd.png)

- $\ell_{\text{eval}}(\cdot, \cdot, \phi^*)$ represents the evaluation loss function, such as cross-entropy loss, which is parameterized by the neural network $ϕ^∗$ that has been optimized from the distilled dataset $D^S$

The data synthesis process primarily determines the quality of the distilled datasets.

# Improved Design Choices

# Limitations of Prior Methods

## Lacking Realism

Training-dependent condensation algorithms for datasets, particularly those employed for large-scale datasets, typically initiate the optimization process using Gaussian noise inputs. This initial choice complicates the optimization process and often
results in the generation of synthetic images that do not exhibit high levels of realism.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/lacking%20realism.png)

## Coarse-grained Matching Mechanism

The Statistical Matching (SM)-based pipeline exhibits two critical drawbacks:

- it does not account for the domain discrepancies among different categories
- it fails to preserve the integrity of category-specific information across the original and condensed samples within each batch.

These limitations result in a coarse-grained matching approach that diminishes the accuracy of the matching process.

## Overly Sharp of Loss Landscape

The optimization objective $L(θ)$ can be expanded through a second-order Taylor expansion. However, earlier training-dependent condensation algorithms neglect to minimize the Frobenius norm of the Hessian matrix H to obtain a flat loss landscape for enhancing its generalization capability through sharpness-aware minimization theory.

## Irrational Hyperparameter Settings

RDED adopts a smoothing LR schedule and uses a reduced batch size for post-evaluation on the full 224×224 ImageNet-1k. These changes, although critical, lack detailed explanations and impact assessments in the existing literature. Our empirical analysis highlights a remarkable impact on performance: absent these modifications, RDED achieves only 25.8% accuracy on ResNet18 with IPC 10. With these modifications, however, accuracy jumps to 42.0%. This work aims to fill the gap by providing the first comprehensive empirical analysis and ablation study on the effects of these and similar improvements in the field.

##  Our Solutions

## Real Image Initialization

- Using real images instead of Gaussian noise for data initialization during the data synthesis phase.

This method significantly improves the realism of the condensed dataset and simplifies the optimization process, thus enhancing the synthesized dataset’s ability to generalize in post-evaluation tests.

- Incorporating considerations of information density and efficiency by employing a training-free condensed dataset (typically via RDED) for initialization at the start of the synthesis process.
  
The cost of transporting from a Gaussian distribution to the original data distribution is higher than using the training-free condensed distribution as the initial reference. This advantage also allows us to reduce the number of iterations needed to achieve results to half of those required by our baseline G-VBSM model, significantly boosting synthesis efficiency.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/real%20image.png)

## Soft Category-Aware Matching

Previous dataset condensation methods based on the Statistical Matching (SM) framework have shown satisfactory results predominantly when the data follows an unimodal distribution (e.g., a single Gaussian).

### Limitation of this method:

Datasets consist of multiple classes with significant variations among their class distributions. Traditional SM-based methods compress data by collectively processing all samples, thus neglecting the differences between classes.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/soft%20category-aware%20matching.png)

-  As shown in the top part, this method enhances information density but also creates a big mismatch between the condensed source distribution $X^S$ and the target distribution $X^T$

The use of a Gaussian Mixture Model (GMM) to effectively approximate any complex distribution to tackle this problem. This solution is theoretically justifiable by the Tauberian Theorem under certain conditions. In light of this, we define two specific approaches to Statistical Matching:

- Given $N$ random samples  $`\{x_i\}_{i=1}^N`$  with an unknown distribution  $`p_{mix}(x)`$.
- Form (1): involves synthesizing $M$ distilled samples $`\{y_i\}_{i=1}^M`$ ,where $`M ≪ N`$, ensuring that the variances and means of both  $`\{x_i\}_{i=1}^N`$  and  $`\{y_i\}_{i=1}^M`$  are consistent.
- Form (2): treats $`p_{mix}(x)`$ as a $GMM$ with $C$ components. For random samples $`\{x_i^j\}_{i=1}^{N_j} \quad \left( \sum_{j} N_j = N \right)`$  within each component $`c_j`$ , we synthesize  $`M_j \quad \left( \sum_{j} M_j = M \right)`$  distilled samples $`\{y_i^j\}_{i=1}^{M_j}`$, where $`M_j ≪ N_j`$ , to maintain the consistency of variances and means between
$`\{x_{j_i}\}_{i=1}^{N_j} \quad \text{and} \quad \{y_{j_i}\}_{i=1}^{M_j}`$

However, our empirical result indicates that exclusive reliance on Form (1) yields a synthesized dataset that lacks sufficient
information density. Consequently, we propose for a hybrid method that effectively integrates Form(1) and Form(2) using a weighted average, which we term soft category-aware matching.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/ema.png)

- $C$ represents the total number of components
- $c_i$ indicates the $i-th$ component within a GMM,
- $α$ is a coefficient for adjusting the balance.

The modified loss function $`L'_{\text{syn}}`$ is designed to 
effectively regulate the information density of $`X^S`$ and to align the distribution of $`X^S`$ with that of $`X^T`$. Operationally, each category in the original dataset is mapped to a distinct component in the GMM framework. Particularly, when $`α`$ = 1, the sophisticated category-aware matching described by
$`L'_{\text{syn}}`$ simplifies to the basic statistical matching defined by $`L_{\text{syn}}`$.

(unfinished)

## Flatness Regularization and EMA-based Evaluation 

These two choices are utilized to ensure flat-loss landscapes during the stages of data synthesis and post-evaluation, respectively.

### sharpness-aware minimization (SAM)

The applicable SAM algorithm aims to solve the following maximum minimization problem:

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/sma.png)

- $`L_S(f_θ)`$, $`ϵ`$, $`ρ`$, and $`θ`$ refer to the loss $`\frac{1}{|S|} \sum_{(x_i, y_i) \sim S} \ell(f_{\theta}(x_i), y_i)`$, the perturbation, the pre-defined flattened region, and the model parameter, respectively.

##

During the data synthesis phase
- The use of sharpness-aware minimization (SAM) algorithms is beneficial for reducing the sharpness of the loss landscape.
- Traditional SAM approaches generally double the computational load due to their two-stage parameter update process. This increase in computational demand is often impractical during data synthesis.

A lightweight flatness regularization approach for implementing SAM during data synthesis is introduced.This method utilizes a teacher dataset, $X_{EMA}^S$, maintained via exponential moving average (EMA). The newly formulated optimization goal aims to foster a flat loss landscape in the following manner:

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/lfr.png)

- $β$ is the weighting coefficient, which is empirically set to 0.99 in our experiments

##### The critical theoretical result is articulated as follows: $`\text{The optimization objective $L_{FR}$ can ensure sharpness-aware minimization within a $`1`$-ball for each point along a straight path between $`X^S`$ and $X_{EMA}^S$.}`$

This indicates that the primary optimization goal of $L_{FR}$ deviates somewhat from that of traditional SAM-based algorithms, which are designed to achieve a flat loss landscape around $X^S$. While both $L^{FR}$ and conventional SAM-based methods are capable of performing sharpness-aware training, our findings unfortunately demonstrate that various SM-based loss functions do not converge to zero. This failure to converge contradicts the basic premise that the first-order term in the Taylor expansion
should equal zero. As a result, we choose to apply flatness regularization exclusively to the logits of the observer model, since the cross-entropy loss for these can more straightforwardly reach zero.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/l'fr.png)

- Softmax(·), $τ$ and $`ϕ`$ represent the softmax operator, the temperature coefficient and the pretrained observer model respectively

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/flatness.png)

##### (top)It is evident that $`L′_{FR}`$ significantly lowers the Frobenius norm of the Hessian matrix relative to standard training, thus confirming its efficacy in pushing a flatter loss landscape.


In post-evaluation, we observe that a method analogous to $L′_{FR}$ employing SAM does not lead to appreciable performance improvements. This result is likely due to the limited sample size of the condensed dataset, which hinders the model’s ability to fully converge post-training, thereby undermining the advantages of flatness regularization. Conversely, the integration of an EMA-updated model as the validated model markedly stabilizes performance variations during evaluations. We term this strategy EMA-based evaluation and apply it across all benchmark experiments.

## Smoothing Learning Rate (LR) Schedule and Smaller Batch Size

To optimize the training with condensed samples, we implement a smoothed LR schedule that moderates the learning rate reduction throughout the training duration. This approach helps avoid early convergence to suboptimal minima, thereby enhancing the model’s generalization capabilities. The mathematical formulation of this schedule is given by:

$$
\mu(i) = \frac{1 + \cos\left(\frac{i\pi}{\zeta N}\right)}{2}
$$

-  $i$ represents the current epoch
-  $N$ is the total number of epochs
-  $µ(i)$ is the learning rate for the $i$-th epoch
-  $ζ$ is the deceleration factor.
-  Notably, a $ζ$ value of 1 corresponds to a typical cosine learning rate schedule, whereas setting $ζ$ to 2 improves performance metrics from 34.4% to 38.7% and effectively moderates loss landscape sharpness during post-evaluation.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/flatness.png)

- The gradient from a random batch in $X^S$ effectively approximates the global gradient.(bottom)
Leveraging this alignment, we can use smaller batch sizes to significantly increase the number of iterations, which helps prevent model under-convergence during post-evaluation.

## Weak Augmentation and Better Backbone Choice

The principal role of these two design decisions is to address the flawed settings in the baseline G-VBSM: the minimum area threshold for cropping during data synthesis was too restrictive, thereby diminishing the quality of the condensed dataset

To rectify this:
- we implement mild augmentations to increase this minimum cropping threshold, thereby improving the dataset condensation’s ability to generalize.
- we substitute the computationally demanding EfficientNet-B0 with more streamlined AlexNet for generating soft labels on ImageNet-1k, a change we refer to as an improved backbone selection. This modification maintains the performance without degradation

# Experiments

To validate the effectiveness of EDC:
- conduct comparative experiments across various datasets, including ImageNet-1k, ImageNet-10, Tiny-ImageNet, CIFAR-100, and CIFAR-10.
- explore cross-architecture generalization and ablation studies on ImageNet-1k.
- All experiments are conducted using 4× RTX 4090 GPUs

Network Architectures:
   - uses ResNet-{18, 50, 101} as our verified models
   - extend our evaluation to include MobileNet-V2
   - explore cross-architecture generalization further with recently advanced backbones such as DeiT-Tiny and Swin-Tiny

Baselines:
   - compare our work with several recent state-of-the-art methods, including SRe2L, G-VBSM, and RDED to assess broader practical impacts.
   - omitted several traditional methods from our analysis. This exclusion is due to their inadequate performance on the large-scale ImageNet-1k and their lesser effectiveness when applied to practical networks such as ResNet, MobileNet-V2, and Swin-Tiny

## Main Results











