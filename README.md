# Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation

# Reference:
1, Shao, S., Zhou, Z., Chen, H., & Shen, Z. (2024). Elucidating the Design Space of Dataset Condensation. arXiv preprint arXiv:2404.13733.

# Elucidate Dataset Condensation (EDC)

- The prohibitively expensive bi-level optimization paradigm limits the effectiveness of traditional dataset distillation methods
- While uni-level optimization has demonstrated remarkable performance on large datasets, it has yet to achieve the benchmark accuracy levels seen with classical methods on datasets like CIFAR-10/100
- The newly developed training-free RDED significantly outperforms previous methods in efficiency and maintains effectiveness, yet it overlooks the potential information loss due to the lack of image optimization
- Some simple but promising techniques (e.g., smoothing the learning rate schedule) that could enhance performance have been underexplored in existing literature.

These drawbacks show the constraints of previous methods in several respects, highlighting the need for a thorough investigation and assessment of these issues. Therefore, Elucidate Dataset Condensation (EDC), which includes a range of detailed
and effective enhancements, is developed to remedy these drawbacks of dataset distillation. 

# Dataset Condensation

Dataset condensation involves generating a synthetic dataset:

$D^S := \{(x_{i}^{S}, y_{i}^{S})\}_{i=1}^{|D^S|}$ 

- consisting of images $X^S$ and labels $Y^S$

designed to be as informative as the original dataset:

$D^T := \{(x_{i}^{T}, y_{i}^{T})\}_{i=1}^{|D^T|}$ 

- includes images $X^T$ and labels $Y^T$

The synthetic dataset $D^S$ is substantially smaller in size than $D^T$($|D^S|$ $≪$ $|D^T|$). The goal of this process is to maintain the
critical attributes of $D^T$ to ensure robust or comparable performance during evaluations on test protocol $P_D$.

![text image](https://github.com/profGiveMeHighGradePLZ/Reading_note.Elucidating-the-Design-Space-of-Dataset-Condensation/blob/main/image/dd.png)

- $\ell_{\text{eval}}(\cdot, \cdot, \phi^*)$ represents the evaluation loss function, such as cross-entropy loss, which is parameterized by the neural network $ϕ^∗$ that has been optimized from the distilled dataset $D^S$

The data synthesis process primarily determines the quality of the distilled datasets

# Improved Design Choices
