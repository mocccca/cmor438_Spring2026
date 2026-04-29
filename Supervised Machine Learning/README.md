## Supervised Machine Learning

Supervised learning is a type of machine learning where the model learns from labeled data — meaning, each training example comes with both the input and the correct output.

The workflow of supervised machine learning is usually:
- Collect labeled data: Each data point has input features and a known output. 
- Split into training and test sets: Usually 70–80% for training, the rest for testing. 
- Choose a model: Based on the problem and data type. 
- Train the model: Minimize the difference between predictions and true outputs using a loss function. 
- Evaluate performance: On the test set using metrics like accuracy, precision, recall (for classification) or RMSE (for regression).

Below listed some common supervised machine learning methods, which are also the ones that are applied in this project:
- The Perceptron
- Linear Regression
- Logistic Regression
- Multilayer Perceptron
- K Nearest Neighbors
- Decision Trees / Regression Trees
- Random Forests
- Other Ensemble Methods, including Boosting

I utilized the supervised machine learning to explore these two research questions:

### Research Question 1: To What Extent Can Personality Predict Workplace Behavior?

This analysis examines whether Big Five personality traits can predict two workplace behavior outcomes:

- **CWB**: Counterproductive Work Behavior, which refers to voluntary employee actions that harm or intend to harm organizations, their members, or clients. 
- **OCB**: Organizational Citizenship Behavior, which refers to voluntary, discretionary actions by employees that are not part of their formal job requirements but promote effective organizational functioning.

The predictors are the five Big Five traits:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

Two types of supervised learning tasks are used here:

1. **Regression models** (*Multilayer Perceptron and Linear Regression*), where CWB and OCB are treated as continuous outcomes.
2. **Classification models** (*Perceptron and Logistics Regression*), where CWB and OCB are divided into low, medium, and high groups.

The goal is to compare whether simple linear models or more complex models better predict workplace behavior from personality.

### Results

##### Classification Results
| Outcome | Model | Accuracy | Precision | Recall | F1 Macro |
|---|---|---:|---:|---:|---:|
| CWB | Logistic Regression | 0.50 | 0.500 | 0.496 | 0.482 |
| CWB | Perceptron | 0.39 | 0.395 | 0.386 | 0.386 |
| OCB | Logistic Regression | 0.40 | 0.352 | 0.378 | 0.351 |
| OCB | Perceptron | 0.38 | 0.365 | 0.372 | 0.365 |

##### Regression Results
| Outcome | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| CWB | Linear Regression | 0.336 | 0.442 | 0.183 |
| CWB | MLP Regression | 0.343 | 0.457 | 0.125 |
| OCB | Linear Regression | 0.612 | 0.757 | 0.143 |
| OCB | MLP Regression | 0.619 | 0.776 | 0.099 |


Results suggest that personality does predict workplace behavior, but the effects were modest overall. CWB was slightly more predictable than OCB, possibly because counterproductive behaviors are more directly linked to stable dispositional tendencies such as low conscientiousness, low agreeableness, and higher neuroticism, whereas OCB may depend more heavily on situational factors such as leadership, organizational climate, team norms, and opportunities to engage in extra-role behavior. Overall, stronger prediction would likely require incorporating variables beyond personality alone.

Linear/logistic models performed better than the neural-network-style models, suggesting that the relationship between Big Five traits and CWB/OCB is probably more linear and simple than highly nonlinear in this dataset.


## Research Question 2: To What Extent Can Narrative Text Data Predict Big Five Personality?

This analysis examines whether participants’ narrative text responses can be used to predict their Big Five personality traits.

The outcomes are now the five Big Five traits:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

The predictors are participants’ written responses to **32 narrative prompts**, with each response averaging around 50-100 words. These narratives were transformed into text features that machine learning models could analyze.

Three main types of supervised learning regression models were used because personality traits are continuous outcomes:

1. **Tree-based models** (*Decision Tree, Random Forest, Gradient Boosting*)  
2. **Distance-based model** (*K-Nearest Neighbors*)  
3. **Support Vector methods** (*Linear SVR and RBF SVR*)  

The goal is to compare which modeling approaches best extract personality-relevant information from narrative language, and whether more flexible nonlinear methods outperform simpler approaches.

### Results

##### Pearson's r Comparison Across Algorithms

| Trait | Decision Tree | Gradient Boosting | KNN | Random Forest | Linear SVR | RBF SVR |
|---|---:|---:|---:|---:|---:|---:|
| Extraversion | -0.0036 | 0.1857 | 0.2233 | 0.2981 | 0.3272 | **0.3667** |
| Agreeableness | -0.0138 | 0.0197 | 0.0046 | 0.0296 | 0.1072 | **0.1234** |
| Conscientiousness | 0.2035 | 0.2730 | 0.1855 | 0.2916 | 0.2948 | **0.3531** |
| Neuroticism | 0.1248 | 0.2551 | 0.2039 | 0.2601 | 0.3672 | **0.3885** |
| Openness | 0.0436 | 0.1591 | 0.1672 | 0.1396 | 0.2600 | **0.3416** |

Narrative text data was able to predict personality traits, but performance varied substantially across traits and algorithms. Pearson's r was mainly assessed to determine the convergent validity of the algorithms with self-report Big Five assessments, as convergent validity is how prediction performance is assessed in psychology.

Across models, the strongest and most consistent prediction was found for:

- **Extraversion**
- **Conscientiousness**
- **Neuroticism**

These traits may be more directly reflected in language through social behavior, emotional expression, and goal-oriented themes.

The weakest prediction was consistently found for:

- **Agreeableness**

This suggests that agreeableness may be less visible in word usage alone or that the narrative prompts were not designed to capture agreeableness directly.

Among all methods tested, **Support Vector Regression (especially non-linear SVR)** performed best overall, outperforming tree-based and distance-based models on all traits. This suggests that support vector methods are particularly effective for high-dimensional text data, and that nonlinear kernels can capture subtle language patterns related to personality differences.

Overall, the findings show that narrative text contains meaningful psychological information, but prediction remains modest, indicating that personality can be inferred from language to some extent, though not perfectly.

