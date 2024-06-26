import pandas as pd
import re
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# Function to parse the log file and extract patient ID, F1 Score, and IoU Score
def parse_log_file(log_file):
    patient_ids = []
    f1_scores = []
    iou_scores = []

    with open(log_file, 'r') as file:
        for line in file:
            match_f1 = re.search(r'(\d+) - ([A-Z0-9]+) - F1 Score: (\d+\.\d+)', line)
            match_iou = re.search(r'(\d+) - ([A-Z0-9]+) - IoU Score: (\d+\.\d+)', line)
            if match_f1:
                patient_ids.append(match_f1.group(2))
                f1_scores.append(float(match_f1.group(3)))
            if match_iou:
                iou_scores.append(float(match_iou.group(3)))

    return patient_ids, f1_scores, iou_scores

path = 'UTAH Test set/log'

# List of log files and their corresponding model names
log_files = [
    (f'{path}/log_MULTIRES_ATT_UNET_MODEL.txt', 'MULTIRES_ATT_UNET_MODEL'),
    (f'{path}/log_ATT_UNET_MODEL.txt', 'ATT_UNET_MODEL'),
    (f'{path}/log_MULTIRES_MODEL.txt', 'MULTIRES_MODEL'),
    (f'{path}/log_UNET_MODEL.txt', 'UNET_MODEL')
]

# Initialize dictionaries to hold the data
data_f1 = {}
data_iou = {}

# Parse each log file and store the results
for log_file, model_name in log_files:
    patient_ids, f1_scores, iou_scores = parse_log_file(log_file)
    data_f1[model_name] = f1_scores
    data_iou[model_name] = iou_scores

# Create DataFrames
df_f1_score = pd.DataFrame(data_f1)
df_iou_score = pd.DataFrame(data_iou)

# Adding patient IDs as the index
df_f1_score['Patient ID'] = patient_ids
df_iou_score['Patient ID'] = patient_ids

# Set Patient ID as index
df_f1_score.set_index('Patient ID', inplace=True)
df_iou_score.set_index('Patient ID', inplace=True)

# Calculate mean and standard deviation for F1 Score and IoU Score
metrics_summary = pd.DataFrame({
    'Model': ['MULTIRES_ATT_UNET_MODEL', 'ATT_UNET_MODEL', 'MULTIRES_MODEL', 'UNET_MODEL'],
    'Mean F1 Score': [df_f1_score[model].mean() for model in df_f1_score.columns],
    'Std F1 Score': [df_f1_score[model].std() for model in df_f1_score.columns],
    'Mean IoU Score': [df_iou_score[model].mean() for model in df_iou_score.columns],
    'Std IoU Score': [df_iou_score[model].std() for model in df_iou_score.columns]
})

# Paired t-test for F1 Scores
f1_ttest_results = {}
for model in df_f1_score.columns:
    if model != 'MULTIRES_ATT_UNET_MODEL':
        t_stat, p_val = stats.ttest_rel(df_f1_score['MULTIRES_ATT_UNET_MODEL'], df_f1_score[model])
        f1_ttest_results[model] = (t_stat, p_val)

# Paired t-test for IoU Scores
iou_ttest_results = {}
for model in df_iou_score.columns:
    if model != 'MULTIRES_ATT_UNET_MODEL':
        t_stat, p_val = stats.ttest_rel(df_iou_score['MULTIRES_ATT_UNET_MODEL'], df_iou_score[model])
        iou_ttest_results[model] = (t_stat, p_val)

# Print the metrics summary
print("Metrics Summary:")
for index, row in metrics_summary.iterrows():
    print(f"Model: {row['Model']}")
    print(f"Mean F1 Score: {row['Mean F1 Score']:.2f}")
    print(f"Std F1 Score: {row['Std F1 Score']:.2f}")
    print(f"Mean IoU Score: {row['Mean IoU Score']:.2f}")
    print(f"Std IoU Score: {row['Std IoU Score']:.2f}")
    print("-----------------------------")

# Print the t-test results for F1 Scores
print("Paired t-test Results for F1 Scores:")
for model, result in f1_ttest_results.items():
    print(f"Model: {model}")
    print(f"T-statistic: {result[0]:}")
    print(f"P-value: {result[1]:}")
    print("-----------------------------")

# Print the t-test results for IoU Scores
print("Paired t-test Results for IoU Scores:")
for model, result in iou_ttest_results.items():
    print(f"Model: {model}")
    print(f"T-statistic: {result[0]:}")
    print(f"P-value: {result[1]:}")
    print("-----------------------------")

# Visualize the distribution of F1 Scores
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_f1_score)
plt.title('Distribution of F1 Scores by Model')
plt.ylabel('F1 Score')
plt.xlabel('Model')
plt.savefig('Distribution of F1 Scores by Model.png', dpi=300)
plt.show()

# Visualize the distribution of IoU Scores
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_iou_score)
plt.title('Distribution of IoU Scores by Model')
plt.ylabel('IoU Score')
plt.xlabel('Model')
plt.savefig('Distribution of IoU Scores by Model.png', dpi=300)
plt.show()


"""
Analysis:

	1.	Mean and Standard Deviation:
	•	The MULTIRES_ATT_UNET_MODEL has the highest mean F1 Score (0.8681) and IoU Score (0.7684), indicating better average performance compared to the other models.
	•	It also has a lower standard deviation, suggesting more consistent performance.
	2.	Statistical Significance (Paired t-tests):
	•	F1 Scores:
	•	The MULTIRES_ATT_UNET_MODEL shows a highly statistically significant improvement over the ATT_UNET_MODEL (p-value < 0.001).
	•	The improvement over the MULTIRES_MODEL and UNET_MODEL is also statistically significant (p-value < 0.05).
	•	IoU Scores:
	•	The MULTIRES_ATT_UNET_MODEL shows a highly statistically significant improvement over the ATT_UNET_MODEL (p-value < 0.001).
	•	The improvement over the MULTIRES_MODEL and UNET_MODEL is also statistically significant (p-value < 0.01).
	3.	Visualization:
	•	The boxplots show the distribution of F1 Scores and IoU Scores for each model, with the MULTIRES_ATT_UNET_MODEL generally achieving higher scores and less variability compared to the other models.

Conclusion:

The MULTIRES_ATT_UNET_MODEL outperforms the other models, particularly the ATT_UNET_MODEL and UNET_MODEL, with statistically significant improvements in both F1 Score and IoU Score. The comparison with MULTIRES_MODEL also shows significant improvement, solidifying the superiority of the MULTIRES_ATT_UNET_MODEL in terms of performance and consistency.
"""

# Calculate the best F1 score and IoU score among the other three models
df_f1_score['Best_Other'] = df_f1_score[['ATT_UNET_MODEL', 'MULTIRES_MODEL', 'UNET_MODEL']].max(axis=1)
df_iou_score['Best_Other'] = df_iou_score[['ATT_UNET_MODEL', 'MULTIRES_MODEL', 'UNET_MODEL']].max(axis=1)

# Calculate the difference between MULTIRES_ATT_UNET_MODEL and the best of the other three models
df_f1_score['Diff_Best_Other'] = df_f1_score['MULTIRES_ATT_UNET_MODEL'] - df_f1_score['Best_Other']
df_iou_score['Diff_Best_Other'] = df_iou_score['MULTIRES_ATT_UNET_MODEL'] - df_iou_score['Best_Other']

# Calculate mean and standard deviation of the differences
f1_mean_diff = df_f1_score['Diff_Best_Other'].mean()
f1_std_diff = df_f1_score['Diff_Best_Other'].std()
iou_mean_diff = df_iou_score['Diff_Best_Other'].mean()
iou_std_diff = df_iou_score['Diff_Best_Other'].std()

# Calculate the percentage of patients where MULTIRES_ATT_UNET_MODEL performs better
f1_better_percentage = (df_f1_score['Diff_Best_Other'] > 0).mean() * 100
iou_better_percentage = (df_iou_score['Diff_Best_Other'] > 0).mean() * 100

print("F1 Score Metrics:")
print(f"Mean difference: {f1_mean_diff:.4f}")
print(f"Standard deviation of difference: {f1_std_diff:.4f}")
print(f"Percentage of patients where MULTIRES_ATT_UNET_MODEL is better: {f1_better_percentage:.2f}%")

print("\nIoU Score Metrics:")
print(f"Mean difference: {iou_mean_diff:.4f}")
print(f"Standard deviation of difference: {iou_std_diff:.4f}")
print(f"Percentage of patients where MULTIRES_ATT_UNET_MODEL is better: {iou_better_percentage:.2f}%")
