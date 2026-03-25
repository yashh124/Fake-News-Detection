import pandas as pd
import matplotlib.pyplot as plt

# LOAD PREVIOUS RESULTS
df = pd.read_csv("models/text/text_model_results.csv")

# ROUND VALUES
df["Accuracy"] = df["Accuracy"].round(4)
df["Precision"] = df["Precision"].round(4)
df["Recall"] = df["Recall"].round(4)
df["F1 Score"] = df["F1 Score"].round(4)

# SORT BY ACCURACY
df = df.sort_values(by="Accuracy", ascending=False)

print(df)

# SAVE CLEAN CSV
df.to_csv("models/text/final_text_results.csv", index=False)

# CREATE TABLE IMAGE
fig, ax = plt.subplots(figsize=(12,6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2,1.2)

plt.title("Fake News Detection Model Comparison", fontsize=14)
plt.savefig("models/text/final_results_table.png")
plt.show()

# LATEX TABLE
latex_table = df.to_latex(index=False)
with open("models/text/final_results_table.tex","w") as f:
    f.write(latex_table)