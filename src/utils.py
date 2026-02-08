import logging
import seaborn as sns
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_class_distribution(df, label_col="label", figsize=(12, 6)):
    """
    Visualize class distribution
    :param df: (pd.DataFrame) : Dataset
    :param label_col: (str) Label column name
    :param figsize: (tuple) Figure size
    :return: plot of class distribution
    """
    plt.figure(figsize=figsize)
    class_counts = df[label_col].value_counts()
    colors = ['#2ecc71', '#e74c3c']

    plt.bar(['HAM', 'SPAM'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Label', fontsize=12)

    # Add value labels on bars
    for i, (label, count) in enumerate(class_counts.items()):
        percentage = (count / len(df)) * 100
        plt.text(i, count + 50, f'{count}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return plt