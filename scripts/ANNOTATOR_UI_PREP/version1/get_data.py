import boto3
import csv
from datetime import datetime


def get_all_mturk_data():
    # Initialize the MTurk client
    mturk = boto3.client(
        "mturk",
        region_name="us-east-1",
        endpoint_url="https://mturk-requester.us-east-1.amazonaws.com",
    )

    # List to store all HIT data
    all_hit_data = []

    # Paginate through all HITs
    paginator = mturk.get_paginator("list_hits")
    for page in paginator.paginate():
        for hit in page["HITs"]:
            hit_id = hit["HITId"]
            # Get HIT details
            hit_details = mturk.get_hit(HITId=hit_id)["HIT"]

            # Get assignments for this HIT
            assignments = []
            place = "list_assignments_for_hit"
            assignment_paginator = mturk.get_paginator(place)
            for assignment_page in assignment_paginator.paginate(HITId=hit_id):
                assignments.extend(assignment_page["Assignments"])

            # Combine HIT details with assignment data
            for assignment in assignments:
                combined_data = {
                    "HITId": hit_id,
                    "HITTypeId": hit_details.get("HITTypeId"),
                    "Title": hit_details.get("Title"),
                    "Description": hit_details.get("Description"),
                    "Keywords": hit_details.get("Keywords"),
                    "Reward": hit_details.get("Reward"),
                    "CreationTime": hit_details.get("CreationTime"),
                    "AssignmentId": assignment.get("AssignmentId"),
                    "WorkerId": assignment.get("WorkerId"),
                    "AssignmentStatus": assignment.get("AssignmentStatus"),
                    "AcceptTime": assignment.get("AcceptTime"),
                    "SubmitTime": assignment.get("SubmitTime"),
                    "Answer": assignment.get("Answer"),
                }
                all_hit_data.append(combined_data)

    return all_hit_data


def save_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


if __name__ == "__main__":
    print("Fetching data from MTurk...")
    all_data = get_all_mturk_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mturk_data_{timestamp}.csv"
    save_to_csv(all_data, filename)
    print(f"Data saved to {filename}")
