import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd

with open("results.json") as f:
    results = json.load(f)


df = pd.DataFrame(results)

df["percent_correct"] = ((df["n_prompts"] - df["n_false"]) / df["n_prompts"]) * 100

# Plotting the data with percent correct prompts
sns.set_theme(style="whitegrid")
# plt.figure(figsize=(14, 8))
sns.barplot(x="category", y="percent_correct", data=df, color="green")

plt.xticks(rotation=90)
plt.xlabel("category")
plt.ylabel("accuracy (%)")
plt.ylim(0, 100)  
plt.title("accuracy of safety classifier on harmful prompts of different categories")

plt.show()



