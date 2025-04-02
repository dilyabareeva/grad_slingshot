import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set font to Helvetica
matplotlib.rcParams['font.family'] = 'Helvetica'

# Read the CSV data from file
df = pd.read_csv('results/benchmark.csv')


df['dataset'] = df['dataset'] + '\n(' + df['task'] + ')'
# Convert relevant columns to numeric
df['manipulated'] = df['manipulated'].astype(bool)
metrics = ['acc1', 'image_retrieval_recall@5']
for metric in metrics:
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

# Determine the appropriate metric per dataset
df['selected_metric'] = df.apply(lambda row: row['image_retrieval_recall@5'] if row['task'] == 'zeroshot_retrieval' else (row['acc1'] if row['task'] == 'zeroshot_classification' else row['lp_acc1']), axis=1)

# Compute differences
non_manipulated = df[df['manipulated'] == False].set_index('dataset')['selected_metric']
manipulated = df[df['manipulated'] == True].set_index('dataset')['selected_metric']
diff = (manipulated - non_manipulated).dropna() * 100

# Merge with task information for sorting
diff_df = diff.to_frame(name='delta').reset_index()
diff_df = diff_df.merge(df[['dataset', 'task']].drop_duplicates(), on='dataset')
diff_df['task_order'] = diff_df['task'].apply(lambda x: 1 if x == 'zeroshot_retrieval' else (2 if x == 'linear_probe' else 0))
diff_df = diff_df.sort_values(by=['task_order'], ascending=[False])

# Plot bar chart
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(diff_df['dataset'], diff_df['delta'], color=['green' if val > 0 else 'blue' for val in diff_df['delta']], edgecolor='black', linewidth=1)

plt.axvline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.xlabel("Change in Score After Manipulation, %", fontsize=20)
#plt.ylabel("Benchmark", fontsize=20)
ax.tick_params(axis='both', labelsize=20)
plt.tight_layout()

# Save as SVG
plt.savefig("results/benchmark.svg", format='svg', dpi=1000, bbox_inches='tight')

plt.show()

