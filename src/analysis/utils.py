import re,csv,json
from get_fleiss_kappas import bool_dict_to_int_list
from displaying_annotations_as_a_probability_distribution import process_categories
from pathlib import Path

def change_mturk_annotation_to_more_readable_form(answer_dict):
    if isinstance(answer_dict, list):
        answer_dict = answer_dict[0] 
    category = process_categories(answer_dict['category'])
    location = "indoors" if bool_dict_to_int_list(answer_dict['location'])==0 else "outdoors"
    type_ = "man-made" if bool_dict_to_int_list(answer_dict['type']) == 0 else "natural"
    total = location+"/" + type_+"/"+category
    return total

#do majority vote and have a separate adjudicator
#send Ken a list of images that have the ties and Ken will adjudicate them
def pick_first_of_the_annotations(midjourney_probs, dalle_probs, flux_probs):
    # Initialize a list of 16 zeros
    result = [0] * 16
    midjourney_max_index = midjourney_probs.index(max(midjourney_probs)) if midjourney_probs else None
    dalle_max_index = dalle_probs.index(max(dalle_probs)) if dalle_probs else None
    flux_max_index = flux_probs.index(max(flux_probs)) if flux_probs else None
    # Assign votes to the most frequently occurring index from each source
    if midjourney_max_index is not None:
        result[midjourney_max_index] += 1
    if dalle_max_index is not None:
        result[dalle_max_index] += 1
    if flux_max_index is not None:
        result[flux_max_index] += 1
    return result

def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists():
            return p
    return here.parent  # fallback

def get_max_indices(arr):
    if not arr:
        return []
    max_val = max(arr)
    max_indices = [i for i, val in enumerate(arr) if val == max_val]
    return max_indices

def extract_number_from_url(modelname, original_or_finalized, url):
    """
    Extracts the numerical identifier from the URL based on the model name and original/finalized status.
    """
    # Correct regex pattern: Ensure proper string concatenation without unnecessary '+'
    pattern = rf'{modelname}_{original_or_finalized}/(\d+)(?:-\d+)?\.\w+$'

    # Search for the pattern in the URL
    match = re.search(pattern, url)

    if match:
        return int(match.group(1))  # Extract and convert to integer
    else:
        return None  # Explicitly return None for debugging



def save_judgements_for_ken_to_csv(judgement_for_ken, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['caption', 'url', 'probs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for caption, (url, probs) in judgement_for_ken.items():
            writer.writerow({
                'caption': caption,
                'url': url,
                'probs': json.dumps(probs)  # Save list as JSON string
            })