from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import json
from displaying_annotations_as_a_probability_distribution import process_human_caption_annotations
from displaying_annotations_as_a_probability_distribution import calculate_probability_distribution
from utils import save_judgements_for_ken_to_csv
from utils import get_max_indices, extract_number_from_url, pick_first_of_the_annotations
from utils import change_mturk_annotation_to_more_readable_form
from helper.helper_functions import(
    normalize_caption,
    clip_probs,
    load_combined_df,
    get_set_of,
    WORKERS,
)

RECREATION_CATEGORY = "outdoors/natural/recreation"


class VLMAnnotationEvaluator:
    def __init__(self, path_config):
        self.captions_filepaths = path_config.captions_filepaths
        self.vlm_filepaths = path_config.vlm_filepaths
        self.caption_to_number_mapping_filepath = path_config.caption_to_number_mapping_filepath

        self.vlm_set = set()
        self.model_names = ['flux', 'dalle', 'midjourney']
        self.annotations = {}
        self.caption_mappings = {}
        self.flux_annotations = {}
        self.dalle_annotations = {}
        self.midjourney_annotations = {}
        self.relabel_count = 0
        self.divergences = {}
        self.original_captions_to_index_mapping = defaultdict(int)
        self.finalized_captions_to_index_mapping = defaultdict(int)
        self.jensen_shannon_divergences={}
        self.judgements_for_ken = {'flux': {}, 'dalle': {}, 'midjourney': {}}
        self.sanity_check_for_vlms = {'flux': {}, 'dalle': {}, 'midjourney': {}}
        self.human_caption_annotations_original = {}
        self.human_caption_annotations_finalized = {}

    def extract_vlm_urls(self,vlm_filepaths):
        """
        Creates a set of URLs that contain the model name and state.
        """
        combined_df= load_combined_df(vlm_filepaths)
        for _, row in combined_df.iterrows():
            url = row['Input.image_url']
            self.vlm_set.add(url)
    
    def store_sanity_check(self, caption, flux_url, flux_probs, dalle_url, dalle_probs, midjourney_url, midjourney_probs):
        self.sanity_check_for_vlms['flux'][caption] = (flux_url, flux_probs)
        self.sanity_check_for_vlms['dalle'][caption] = (dalle_url, dalle_probs)
        self.sanity_check_for_vlms['midjourney'][caption] = (midjourney_url, midjourney_probs)


    def store_judgement_for_ken(self, caption, flux_url, flux_probs, dalle_url, dalle_probs, midjourney_url, midjourney_probs):
        if len(get_max_indices(flux_probs)) > 1:
            self.judgements_for_ken['flux'][caption] = (flux_url, flux_probs)
        if len(get_max_indices(dalle_probs)) > 1:
            self.judgements_for_ken['dalle'][caption] = (dalle_url, dalle_probs)
        if len(get_max_indices(midjourney_probs)) > 1:
            self.judgements_for_ken['midjourney'][caption] = (midjourney_url, midjourney_probs)


    def generate_vlm_annotations(self,vlm_files):
        combined_df= load_combined_df(vlm_files)
        for _, row in combined_df.iterrows():
            workerID = row['WorkerId']
            if workerID == "ASSIGNMENT_ID_NOT_AVAILABLE":
                continue
            if workerID not in WORKERS:
                continue
            url = row['Input.image_url']
            answer_dict = json.loads(row['Answer.taskAnswers'])[0]
            total = change_mturk_annotation_to_more_readable_form(answer_dict)
            if "flux" in url:
                self.flux_annotations.setdefault(url, []).append(total)
            elif "dalle" in url:
                self.dalle_annotations.setdefault(url, []).append(total)
            elif "midjourney" in url:
                self.midjourney_annotations.setdefault(url, []).append(total)

    def find_url_by_index(self, modelname,original_or_finalized, target_index ):
        """
        Searches for the URL in vlm_set that contains the target_index.
        """
        for url in self.vlm_set:
            extracted_index = extract_number_from_url(modelname,original_or_finalized, url)
            if extracted_index == target_index:
                return url
        return None

    def generate_indexes_for_captions(self,filepaths):
        combined_df = load_combined_df(filepaths)
        for index, row in combined_df.iterrows():
            self.original_captions_to_index_mapping[normalize_caption(row['Original Sentence'])] = index
            self.finalized_captions_to_index_mapping[normalize_caption(row['Finalized sentence'])] = index

    def calculate_jensen_shannon_divergence(self,human_probs, llm_probs,caption):
        # Ensure no zero probabilities to avoid issues in KL divergence calculation
        human_probs = clip_probs(human_probs)
        llm_probs = clip_probs(llm_probs)

        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(human_probs, llm_probs)
        self.jensen_shannon_divergences[caption] = js_div

    def save_all_judgements_for_ken(self):
        for model in self.model_names:
            save_judgements_for_ken_to_csv(
                self.judgements_for_ken[model],
                f"{model}_judgements_for_ken2.csv"
            )
        print("Judgements for Ken saved successfully.")

    def save_all_sanity_checks(self):
        for model in self.model_names:
            save_judgements_for_ken_to_csv(
                self.sanity_check_for_vlms[model],
                f"{model}_sanity_check.csv"
            )
        print("Judgements for Ken saved successfully.")


    def _load_all_data(self):
        original_captions_set = get_set_of(self.caption_to_number_mapping_filepath, "Original Sentence")
        self.human_caption_annotations_original, self.human_caption_annotations_finalized = process_human_caption_annotations(
            original_captions_set, self.captions_filepaths
        )
        self.extract_vlm_urls(self.vlm_filepaths)
        self.generate_vlm_annotations(self.vlm_filepaths)
        self.generate_indexes_for_captions(self.caption_to_number_mapping_filepath)


    def _analyze_single_caption_set(self, caption_annotations, is_finalized):
        count_of_relabeling = 0
        for caption in caption_annotations:
            normalized_caption = normalize_caption(caption)
            if is_finalized:
                index_for_caption = self.finalized_captions_to_index_mapping.get(normalized_caption, None)
            else:
                index_for_caption = self.original_captions_to_index_mapping.get(normalized_caption, None)

            if index_for_caption is None:
                continue

            flux_url = self.find_url_by_index("flux", "finalized" if is_finalized else "original", index_for_caption)
            dalle_url = self.find_url_by_index("dalle", "finalized" if is_finalized else "original", index_for_caption)
            midjourney_url = self.find_url_by_index("midjourney", "finalized" if is_finalized else "original", index_for_caption)

            human_probs = calculate_probability_distribution(caption_annotations[caption])
            flux_probs = calculate_probability_distribution(self.flux_annotations[flux_url])
            temp = self.midjourney_annotations[midjourney_url]
            for i, element in enumerate(temp):
                if element == RECREATION_CATEGORY:
                    count_of_relabeling += 1
                    temp[i] = 'outdoors/natural/other_unclear'
            midjourney_probs = calculate_probability_distribution(temp)
            dalle_probs = calculate_probability_distribution(self.dalle_annotations[dalle_url])

            self.store_judgement_for_ken(caption, flux_url, flux_probs, dalle_url, dalle_probs, midjourney_url, midjourney_probs)
            self.store_sanity_check(caption, flux_url, flux_probs, dalle_url, dalle_probs, midjourney_url, midjourney_probs)

            vlm_probs = pick_first_of_the_annotations(midjourney_probs, dalle_probs, flux_probs)
            self.calculate_jensen_shannon_divergence(human_probs, vlm_probs, caption)

        return count_of_relabeling

        
    def analyze_image_annotations(self):
        self._load_all_data()
        count_relabel_original = self._analyze_single_caption_set(self.human_caption_annotations_original, is_finalized=False)
        count_relabel_finalized = self._analyze_single_caption_set(self.human_caption_annotations_finalized, is_finalized=True)
        print("Number of relabeling (original):", count_relabel_original)
        print("Number of relabeling (finalized):", count_relabel_finalized)
        self.save_all_judgements_for_ken()
        self.save_all_sanity_checks()
        return self.jensen_shannon_divergences