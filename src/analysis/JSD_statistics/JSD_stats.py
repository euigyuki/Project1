import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
import numpy as np

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return (mean - h, mean + h)

def bonferroni_correction(p_value, num_comparisons):
    corrected_p = min(p_value * num_comparisons, 1.0)
    return corrected_p

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def compute_jsd_statistics(first,second,third):
    # Load Data
    original_df = pd.read_csv(first)
    finalized_df = pd.read_csv(second)
    output_file = third

    jsd_col = 'Jensen-Shannon Divergence'
    if jsd_col not in original_df.columns or jsd_col not in finalized_df.columns:
        raise ValueError(f"Column '{jsd_col}' not found in one of the files.")

    # Extract Data
    original_jsd = original_df[jsd_col].dropna()
    finalized_jsd = finalized_df[jsd_col].dropna()

    # Compute Statistics
    original_mean = original_jsd.mean()
    finalized_mean = finalized_jsd.mean()
    combined_mean = pd.concat([original_jsd, finalized_jsd]).mean()

    original_std = original_jsd.std()
    finalized_std = finalized_jsd.std()

    original_ci = compute_confidence_interval(original_jsd)
    finalized_ci = compute_confidence_interval(finalized_jsd)

    t_stat, p_value = stats.ttest_ind(original_jsd, finalized_jsd, equal_var=False)
    effect_size = cohens_d(original_jsd, finalized_jsd)
    num_comparisons = 1  # Change to actual number of tests if performing multiple

    bonferroni_p = bonferroni_correction(p_value, num_comparisons)

    # Print Results
    print(f"Mean JSD - Original Captions: {original_mean:.4f}")
    print(f"Mean JSD - Finalized Captions: {finalized_mean:.4f}")
    print(f"Overall Mean JSD: {combined_mean:.4f}")
    print(f"Welch's t-test p-value: {p_value:.4f}")
    print(f"Cohen's d (Effect Size): {effect_size:.4f}")
    print(f"Bonferroni-corrected p-value: {bonferroni_p:.4f}")


    # Prepare Results DataFrame
    results_df = pd.DataFrame({
        'Metric': [
            'Original Mean JSD', 'Finalized Mean JSD', 'Overall Mean JSD',
            'Original Std Dev', 'Finalized Std Dev',
            'Original 95% CI', 'Finalized 95% CI',
            "Welch's t-test p-value", "Bonferroni-corrected p-value", 
            "Cohen's d (Effect Size)"
        ],
        'Value': [
            original_mean, finalized_mean, combined_mean,
            original_std, finalized_std,
            f"{original_ci[0]:.4f} - {original_ci[1]:.4f}",
            f"{finalized_ci[0]:.4f} - {finalized_ci[1]:.4f}",
            p_value, bonferroni_p, effect_size
        ]
    })

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save Results
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


@dataclass
class PathConfig:
    #input paths
    original_captions_jsds: Path
    ablated_captions_jsds: Path
    images_jsds_original_captions: Path
    images_jsds_ablated_captions: Path
    #output paths
    output_csv_sentences_path: Path
    output_csv_images_path: Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def main():
    path_config = PathConfig(
        #input paths
        original_captions_jsds=DATA_DIR / "results" / "original_caption_to_jensenshannon_divergences.csv",
        ablated_captions_jsds=DATA_DIR / "results" / "finalized_caption_to_jensenshannon_divergences.csv",
        images_jsds_original_captions=DATA_DIR / "js_divergence" / "image_to_jensenshannon_divergences_original_captions.csv",
        images_jsds_ablated_captions=DATA_DIR / "js_divergence" / "image_to_jensenshannon_divergences_ablated_captions.csv",
        #output path
        output_csv_sentences_path=DATA_DIR / "results" / "JSD_statistics_sentences.csv",
        output_csv_images_path=DATA_DIR / "results" / "JSD_statistics_images.csv"
    )
    compute_jsd_statistics(path_config.original_captions_jsds,path_config.ablated_captions_jsds, path_config.output_csv_sentences_path)
    compute_jsd_statistics(path_config.images_jsds_original_captions, path_config.images_jsds_ablated_captions, path_config.output_csv_images_path)

if __name__ == '__main__':
    main()
