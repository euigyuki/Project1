from collections import defaultdict
import pandas as pd

class ClassificationMapper:
    def __init__(self, finalized_captions_filepath):
        self.verb_to_classification = defaultdict(str)
        self.sentences_to_classification = defaultdict(str)
        self.processed_sentence_to_classification = defaultdict(str)
        self.sentences_to_verbs = defaultdict(str)
        self.verbs_to_original = defaultdict(list)
        self.verbs_to_processed = defaultdict(list)
        self._load_data(finalized_captions_filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            classification = "/".join(
                str(val) for val in [row['q1'], row['q2'], row['q3'], row['q4']] if pd.notna(val)
            )
            verb = row['Verb']
            sentence = normalize_caption(row['Sentence'])
            finalized = normalize_caption(row['Finalized Sentence'])

            self.verb_to_classification[verb] = classification
            self.sentences_to_verbs[sentence] = verb
            self.sentences_to_verbs[finalized] = verb

            self.verbs_to_original[verb].append(sentence)
            self.verbs_to_processed[verb].append(finalized)

            self.sentences_to_classification[sentence] = classification
            self.processed_sentence_to_classification[finalized] = classification

    def get_classification(self, sentence):
        sentence = normalize_caption(sentence)
        return (self.sentences_to_classification.get(sentence) or 
                self.processed_sentence_to_classification.get(sentence))

    def get_verb(self, sentence):
        result = self.sentences_to_verbs.get(normalize_caption(sentence), '')
        return result
    
    @staticmethod
    def create_location_only_classification(answer_list):
        """
        Extracts only the location (indoor/outdoor) labels from the answer list.
        """
        merged = {key: value for d in answer_list for key, value in d.items()}
        indoors_outdoors = merged.get('location', {})
        return indoors_outdoors