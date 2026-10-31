# Intelligent Support Ticket Triage

## 1. Project Overview

Intelligent Support Ticket Triage is a machine learning project that uses Natural Language Processing (NLP) to automatically classify customer support messages into predefined categories.

The aim is to reduce the manual effort involved in reviewing and categorising customer support tickets by developing a model that can analyse written messages, identify patterns in the text, and predict the most appropriate support category.

The system classifies incoming messages into six categories:

- Billing
- Cancellation
- Technical
- API
- Complaint
- Upgrade

For each message, the model produces a predicted category and an estimated confidence score. A confidence-based review mechanism can flag uncertain predictions for human review.

The project was developed using Python, Scikit-learn, Pandas and Hugging Face Datasets.

## 2. Business Problem

Customer support teams receive large volumes of messages relating to different issues, including payment problems, subscription cancellations, technical errors and complaints.

Manually reviewing every message and deciding which department should handle it can be time-consuming and inefficient.

Incorrect categorisation can also result in tickets being directed to the wrong team, potentially delaying customer support.

This project explores how machine learning can assist with the initial classification of customer support messages.

By learning patterns from previously labelled support tickets, the model can predict the likely category of a new message and provide a suggested classification.

The intended benefits include:

- Reducing repetitive manual ticket categorisation.
- Supporting faster initial ticket assessment.
- Improving consistency in how messages are categorised.
- Identifying uncertain predictions that may require human judgement.

The system is designed as a classification and decision-support tool. It does not independently resolve customer enquiries or automatically send tickets to support departments.

## 3. Dataset and Limitations

### Dataset source

The project uses the Support Ticket Router 12K Cleaned dataset, obtained through Hugging Face Datasets.

Dataset: https://huggingface.co/datasets/cngchis/Support-Ticket-Router-12K-Cleaned

The dataset contains approximately 11,693 labelled support-ticket messages divided into three separate subsets:

| Dataset split | Number of examples |
|---|---:|
| Training | 9,354 |
| Validation | 1,169 |
| Test | 1,170 |
| **Total** | **11,693** |

Each example contains three fields:

- `text`: The customer support message.
- `label`: The correct support-ticket category.
- `length`: The dataset's message-length field.

### Classification categories

The model was trained to identify six categories:

| Category | Description |
|---|---|
| Billing | Payment, invoice and charging-related issues |
| Cancellation | Requests relating to cancelling subscriptions or services |
| Technical | General technical issues |
| API | API-related problems and errors |
| Complaint | Customer dissatisfaction and complaints |
| Upgrade | Subscription and plan upgrade enquiries |

### Data limitations

The dataset is synthetic rather than a collection of genuine customer support conversations.

This is an important limitation because synthetic messages may contain more consistent language patterns than real customer enquiries.

Real support messages may include:

- Spelling mistakes and grammatical errors.
- Abbreviations and informal language.
- Multiple issues within one message.
- Ambiguous requests.
- Missing context.
- Unusual technical terminology.

The model's performance on synthetic data does not guarantee equivalent performance on real customer support messages.

The dataset also contains predefined labels, meaning the model learns the category definitions and annotation patterns present in the supplied data.

Further testing on representative, appropriately anonymised real-world data would be required before considering production deployment.

## 4. Data Exploration and Preparation

Before training the model, I carried out exploratory data analysis to understand the dataset's structure and quality.

Using Python and Pandas, I investigated:

- Dataset dimensions and column names.
- Missing values.
- Exact duplicate records.
- Distribution of support-ticket categories.
- Message-length statistics.
- Example messages from each category.

### Data quality findings

The training dataset contained:

- 9,354 examples.
- Zero missing text values.
- Zero missing labels.
- Zero exact duplicate rows.
- Zero duplicate ticket texts within the training split.

The six categories were also relatively balanced, with each representing approximately 16–17% of the training data.

This meant there was no substantial class imbalance requiring an initial resampling strategy.

### Dataset preparation

The dataset already provided separate training, validation and test splits.

These were retained for the machine learning workflow:

- Training data was used to fit the model.
- Validation data was used to compare model approaches and assess development decisions.
- Test data was reserved for final evaluation.

The input messages (`text`) were separated from their corresponding target labels (`label`) to support supervised learning.

The dataset was checked for missing values across all three splits before model development.

## 5. Model Selection and Machine Learning Approach

This project uses supervised machine learning for text classification.

Supervised learning involves training a model using examples where the correct output is already known.

In this project, the input is a customer support message and the expected output is one of six support-ticket categories.

### TF-IDF: Text Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was selected to convert written messages into numerical features that a machine learning algorithm can process.

In simple terms, TF-IDF assigns numerical importance to words and phrases based on their frequency within a message and how common they are across the dataset.

For example, terms such as:

- Invoice
- Payment
- Subscription
- Cancellation
- Endpoint

may provide useful information when distinguishing between support-ticket categories.

The word-level TF-IDF configuration used:

- Lowercase text processing.
- Unigrams and bigrams (individual words and pairs of consecutive words).
- A minimum document frequency of two.
- A maximum vocabulary size of 50,000 features.
- Sublinear term-frequency scaling.

### Logistic Regression: Classification

Logistic Regression was selected as the classification algorithm.

Despite its name, Logistic Regression can be used for multiclass classification.

It learns relationships between numerical text features and the labelled support-ticket categories.

When a new message is provided, the trained classifier calculates scores for the available categories and selects the category with the highest estimated probability.

It was chosen because it is:

- Efficient to train.
- Suitable for text-classification tasks.
- A useful baseline for supervised NLP.
- Relatively straightforward to evaluate and interpret.

### Pipeline architecture

Scikit-learn's Pipeline functionality was used to combine text feature extraction and classification into one reusable model.

This ensures that new messages undergo the same text-processing steps before classification.

## 6. Experimentation: Character-Level TF-IDF

After developing the initial word-based model, I experimented with adding character-level TF-IDF features.

The purpose was to investigate whether additional character-level information could improve classification performance.

Word-level features identify individual words and phrases.

Character-level features examine smaller sequences of characters within words. These can help capture useful patterns in technical terminology, word variations and spelling differences.

I combined both approaches using Scikit-learn's FeatureUnion.

The resulting architecture contained:

1. Word-level TF-IDF.
2. Character-level TF-IDF.
3. Logistic Regression.

The character-level configuration used character n-grams ranging from three to five characters.

The combined features were then passed into the same Logistic Regression classification algorithm.

This experiment was designed to test whether combining different types of textual information improved predictive performance.

## 7. Model Comparison Results

Two approaches were compared using the validation dataset.

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Word-level TF-IDF + Logistic Regression | 98.80% | 0.9881 |
| Word + character TF-IDF + Logistic Regression | 99.49% | 0.9949 |

The combined model improved validation accuracy by approximately 0.69 percentage points.

Its macro F1 score also improved from 0.9881 to 0.9949.

Based on these validation results, the combined word and character feature approach was selected as the preferred model.

This experiment demonstrated the value of comparing alternative feature-extraction approaches rather than assuming the initial model was the best option.

## 8. Final Model Evaluation

The selected combined model was trained using the training dataset and evaluated using both validation and held-out test data.

### Final results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 99.49% | 99.49% |
| Macro F1 | 0.9949 | 0.9949 |

The test dataset contained 1,170 examples.

The rounded test accuracy of 99.49% corresponds to approximately 1,164 correctly classified messages and six incorrect predictions.

### Per-category performance

The final test classification report showed:

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| API | 0.99 | 0.99 | 0.99 |
| Billing | 0.99 | 0.99 | 0.99 |
| Cancellation | 0.99 | 0.99 | 0.99 |
| Complaint | 1.00 | 1.00 | 1.00 |
| Technical | 0.99 | 0.99 | 0.99 |
| Upgrade | 1.00 | 1.00 | 1.00 |

### Evaluation metrics

Several evaluation metrics were used:

- Accuracy: Measures the proportion of all predictions that were correct.
- Precision: Measures how often predictions for a particular category were correct.
- Recall: Measures how many actual examples of a category were correctly identified.
- F1-score: Combines precision and recall into a single metric.
- Macro F1: Calculates the average F1-score across all categories, giving each category equal importance.

A confusion matrix was also generated to identify which categories were most frequently confused.

### Evaluation limitations

Although the model achieved strong results, the evaluation was performed using a synthetic dataset.

The results should therefore be interpreted as performance on the supplied dataset rather than evidence of equivalent real-world performance.

The held-out test set was used for final evaluation and should not be repeatedly used to guide further model tuning.

## 9. Error Analysis

Error analysis was carried out to understand why the model incorrectly classified certain messages.

The initial word-based model produced ten incorrect predictions on the test dataset.

Some examples involved overlapping categories:

- Billing versus cancellation.
- API versus technical.
- Billing versus upgrade.

For example, a message stating that a customer wanted to cancel a subscription but had still been charged could reasonably contain information relating to both billing and cancellation.

The model's predictions were examined alongside their estimated probabilities and the second-highest predicted category.

This helped identify situations where the model was uncertain between two possible categories.

Error analysis demonstrated that a high overall accuracy score does not eliminate the need to investigate individual classification mistakes.

## 10. Confidence-Based Human Review

A confidence-based review mechanism was developed to identify predictions that may require human attention.

The system examines the highest estimated class probability produced by the classifier.

A provisional threshold of 70% was initially evaluated using the validation data from the original word-based model.

### Initial threshold experiment

| Confidence threshold | Tickets flagged for review | Errors among tickets not flagged |
|---|---:|---:|
| 50% | 37 (3.2%) | 4 |
| 60% | 66 (5.6%) | 2 |
| 70% | 132 (11.3%) | 0 |
| 80% | 265 (22.7%) | 0 |
| 90% | 586 (50.1%) | 0 |
| 95% | 864 (73.9%) | 0 |

At the initial 70% threshold, 132 of 1,169 validation tickets were flagged for human review.

No incorrect predictions were observed among the 1,037 tickets that were not flagged.

The threshold was selected as a provisional development setting because it demonstrated a useful trade-off between review workload and observed classification errors on that validation split.

**Important:** This initial threshold experiment was conducted using the original word-based model. The threshold must be re-evaluated using the final combined model before being treated as its review setting.

### Human-review workflow

The prediction tool displays:

- The highest-scoring predicted category.
- Its estimated probability.
- The second-highest predicted category.
- Its estimated probability.
- A suggested review action.

Predictions below the configured threshold are flagged for human review.

The mechanism is intended to support human oversight rather than allow the model to make unverified routing decisions.

Model confidence is not a guarantee of correctness. A model can make incorrect predictions even when its estimated probability is high.

Further probability calibration and evaluation on representative real-world data would be needed before production use.

## 11. Running the Project

### Prerequisites

- Python 3.13 (the development environment used for this project).
- Visual Studio Code.
- Internet access for the initial dataset download.
- The Python packages listed in `requirements.txt`.

### Installation

Clone or download the project and open its root folder in Visual Studio Code.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

If Python is not available through PATH, use the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run the project

The following commands assume the terminal is open in the main project directory.

**1. Inspect the dataset**

```powershell
.\.venv\Scripts\python.exe src/inspect_data.py
```

**2. Explore the dataset**

```powershell
.\.venv\Scripts\python.exe src/explore_data.py
```

**3. Prepare and inspect the data splits**

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py
```

**4. Train the original baseline model**

```powershell
.\.venv\Scripts\python.exe src/train.py
```

**5. Analyse model errors**

```powershell
.\.venv\Scripts\python.exe src/analyse_errors.py
```

**6. Compare model approaches**

```powershell
.\.venv\Scripts\python.exe src/compare_models.py
```

**7. Train and evaluate the combined model**

```powershell
.\.venv\Scripts\python.exe src/final_train.py
```

**8. Run the prediction interface**

```powershell
.\.venv\Scripts\python.exe src/predict.py
```

The prediction interface allows users to enter a support-ticket message and receive a predicted category, estimated confidence and suggested review action.

The model is saved locally after training, so it can be loaded for predictions without retraining each time.

## 12. Project Structure

```text
intelligent-support-triage/
│
├── .venv/
│
├── data/
│
├── models/
│   ├── ticket_classifier.joblib
│   └── ticket_classifier_combined.joblib
│
├── reports/
│   ├── confusion_matrix.png
│   ├── misclassified_tickets.csv
│   ├── error_analysis.csv
│   ├── model_comparison.csv
│   ├── final_test_report.csv
│   └── final_confusion_matrix.csv
│
├── src/
│   ├── inspect_data.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── analyse_errors.py
│   ├── compare_models.py
│   ├── final_train.py
│   ├── predict.py
│   └── review.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

The files listed in the reports and project structure sections represent the intended project outputs. Some are generated by different development scripts.

## 13. Future Improvements

Potential future development includes:

- Evaluating the model on representative, appropriately anonymised real customer support messages.
- Investigating category overlap and ambiguous ticket labels.
- Re-evaluating and calibrating confidence thresholds for the combined model.
- Experimenting with additional NLP and machine learning approaches.
- Developing a more user-friendly prediction interface.
- Deploying the trained model using AWS services such as Amazon S3 and AWS Lambda.
- Integrating the classification system with a ticket-management workflow.
- Monitoring model performance and data drift after deployment.

## 14. Conclusion

This project demonstrates the development of a supervised machine learning system for automated customer support ticket classification.

The development process involved dataset exploration, data preparation, feature extraction, model training, model comparison, evaluation and error analysis.

By experimenting with combined word-level and character-level TF-IDF features, I improved validation accuracy from approximately 98.80% to 99.49%.

I also explored confidence-based human review to investigate how model uncertainty could be incorporated into a more responsible ticket-classification workflow.

The project provided practical experience across the machine learning development lifecycle and established a foundation for further experimentation and cloud deployment.

**Technologies:** Python | Pandas | Scikit-learn | NLP | TF-IDF | Logistic Regression | Hugging Face Datasets | Joblib | AWS (planned extension)
