import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images')
os.makedirs(output_dir, exist_ok=True)

def save_table_image(df, title, filename, col_widths=None):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.axis('off')
    
    # Create table
    table = plt.table(cellText=df.values,
                      colLabels=df.columns,
                      loc='center',
                      cellLoc='center',
                      colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')
            
    plt.title(title, pad=20, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {filename}")

# 1. Hoax Detection Performance
data_hoax = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ECE'],
    'Score': ['99.0%', '100.0%', '98.1%', '99.05%', '0.007 (well-calibrated)']
}
df_hoax = pd.DataFrame(data_hoax)
save_table_image(df_hoax, 'Hoax Detection Performance (Standard Test Set)', 'hoax_detection_table.png')

# 2. Summarization Performance
data_rouge = {
    'Model': ['TextRank (TF-IDF)', 'LexRank (TF-IDF)', 'Gemini (Abstractive)'],
    'ROUGE-1': ['0.3938 ± 0.27', '0.2789 ± 0.09*', '0.4520 ± 0.10*'],
    'ROUGE-2': ['0.3193 ± 0.31', '0.0856 ± 0.05*', '0.3850 ± 0.07*'],
    'ROUGE-L': ['0.3718 ± 0.28', '0.2098 ± 0.08*', '0.4310 ± 0.09*']
}
df_rouge = pd.DataFrame(data_rouge)
save_table_image(df_rouge, 'Summarization Performance (ROUGE Scores)', 'summarization_table.png', col_widths=[0.3, 0.2, 0.2, 0.2])

# 3. Knowledge Graph Verification
data_kg = {
    'Mode': ['Extractive (TextRank)', 'Abstractive (unconstrained)', 'Abstractive (KG-constrained)'],
    'Verification Rate': ['94.8%', '72.4%*', '91.8%*'],
    'Hallucination-Free': ['80.0%', '61.3%*', '89.5%*']
}
df_kg = pd.DataFrame(data_kg)
save_table_image(df_kg, 'Hallucination Prevention Performance', 'verification_table.png', col_widths=[0.4, 0.3, 0.3])

# 4. Out-of-Distribution (OOD) Evaluation
data_ood = {
    'Test Type': ['Standard', 'Adversarial', 'Domain Shift'],
    'F1-Score': ['99.05%', '97.44%', '92.3%'],
    'Notes': ['In-distribution test set', 'Clickbait cues removed', 'Different news sources']
}
df_ood = pd.DataFrame(data_ood)
save_table_image(df_ood, 'Hoax Detection: Out-of-Distribution Evaluation', 'ood_evaluation_table.png', col_widths=[0.2, 0.2, 0.6])

print("All table images generated successfully.")
