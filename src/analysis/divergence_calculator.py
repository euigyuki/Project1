from Project1.scripts.helper.helper_functions import clip_probs,categories_to_num_16
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.special import rel_entr


class DivergenceCalculator:
    @staticmethod
    def calculate_probability_distribution(annotations):
        total = [0] * 16
        for annotation in annotations:
            index = categories_to_num_16[annotation]
            total[index] += 1
        return total

    @staticmethod
    def calculate_jensen_shannon_divergence(human_annotations, llm_annotations):
        divergences = {}
        for caption in human_annotations:
            if caption in llm_annotations:
                prev_human_probs = DivergenceCalculator.calculate_probability_distribution(human_annotations[caption])
                prev_llm_probs = DivergenceCalculator.calculate_probability_distribution(llm_annotations[caption])
                human_probs = clip_probs(prev_human_probs)
                llm_probs = clip_probs(prev_llm_probs)
                js_div = jensenshannon(human_probs, llm_probs)
                kl_div_human_llm = np.sum(rel_entr(np.array(human_probs),np.array(llm_probs)))
                kl_div_llm_human = np.sum(rel_entr(np.array(llm_probs),np.array(human_probs)))
                divergences[caption] = {
                "kl_div_human_llm": kl_div_human_llm,
                "kl_div_llm_human": kl_div_llm_human,
                "js_div": js_div,
                "human_probs": prev_human_probs,
                "llm_probs": prev_llm_probs
            }
        return divergences
    
    @classmethod
    def calculate_all(cls, human_annotations, llm_annotations):
        results = {}
        for split in ['original', 'finalized']:
            results[split] = cls.calculate_jensen_shannon_divergence(
                human_annotations[split],
                llm_annotations[split]
            )
        return results

