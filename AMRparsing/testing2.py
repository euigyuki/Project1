import penman

def find_parent_concepts_and_roles(amr_string, location_arguments):
    tree = penman.parse(amr_string)
    
    prefix_to_concept = {}
    location_edges = []
    
    # Populate the prefix-to-concept mapping
    for path, branch in tree.walk():
        role, target = branch
        if role == '/': #concept definition
            prefix_to_concept[tuple(path)] = target
    
    for path, branch in tree.walk():
        role, target = branch

        
        temp = path[:-1] + (0,)
        parent_prefix = tuple(temp)
        parent_concept = prefix_to_concept.get(parent_prefix, None)
        
        # Check if the role is in location_arguments for the parent concept
        if parent_concept and parent_concept in location_arguments:
            role_number = role.replace(':ARG', '')
            if role_number in location_arguments[parent_concept]:
                location_edges.insert(0,path)
    
    return location_edges, prefix_to_concept


# Example Usage
def main():
    amr_string = "(b / believe-01 :ARG0 (g / girl) :ARG1 (s / sit-01 :ARG1 (c / cat) :ARG2 (m / mat)))"
    location_arguments = {
        "sit-01": {"2"},  # Example: sit-01 has a location role defined as ARG2
    }
    
    location_edges, prefix_to_concept = find_parent_concepts_and_roles(amr_string, location_arguments)
    
    print("Prefix-to-Concept Mapping:")
    for prefix, concept in prefix_to_concept.items():
        print(f"{prefix}: {concept}")
    
    print("\nLocation Edges:")
    for parent_concept, role, target in location_edges:
        print(f"Parent: {parent_concept}, Role: {role}, Target: {target}")

if __name__ == "__main__":
    main()
