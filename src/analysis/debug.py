import csv
import ast

filename = 'midjourney_sanity_check.csv'
filename2 = 'flux_sanity_check.csv'
filename3 = 'dalle_sanity_check.csv'

filename4 = "dalle_judgements_for_ken2.csv"
filename5 = "midjourney_judgements_for_ken2.csv"
filename6 = "flux_judgements_for_ken2.csv"

def filter_probs_sum4(filename):
    count =0
    total = 0
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            total += 1  
            prob_list = ast.literal_eval(row[2])  
            if sum(prob_list) == 4:
                count += 1
            else:
                print(f"Row with sum != 4: {row}")
    print(f"Total matches found: {count}, Total rows: {total}")

if __name__ == "__main__":
    filter_probs_sum4(filename)
    filter_probs_sum4(filename2)
    filter_probs_sum4(filename3)
    filter_probs_sum4(filename4)
    filter_probs_sum4(filename5)
    filter_probs_sum4(filename6)