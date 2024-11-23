import penman


def remove_location_info(amr_string):
    print("amr string", amr_string)
    graph = penman.decode(amr_string)
    triples = graph.triples.copy()
    print("triples", triples)
    print("length of triples", len(triples))
    instances = graph.instances()
    print("instances", instances)
    print("length of instances", len(instances))
    edges = graph.edges()
    print("edges", edges)
    print("length of edges", len(edges))
    top = graph.top
    print("top", top)
    triples = instances + edges
    original_graph = penman.Graph(triples, top=top)
    print("original graph encoding", penman.encode(original_graph))

    # print("original graph", original_graph)
    return None

    # Step 1: Remove location-related edges
    non_location_triples = []
    for edge in edges:
        print("edge", edge)
        if edge[1] != ":location":  # and triple[0]!= top and triple[2]!= top:
            non_location_triples.append(edge)
        else:
            print("location triple", edge)

    print("non location triples", non_location_triples)

    # Step 2: Find reachable nodes using BFS
    reachable = set([top])
    print("reachable", reachable)
    queue = [top]
    while queue:
        node = queue.pop(0)
        print("node", node)
        for s, _, t in non_location_triples:
            # print("s", s, "t", t)
            # print()
            if s == node and t not in reachable:
                reachable.add(t)
                queue.append(t)
            elif t == node and s not in reachable:
                reachable.add(s)
                queue.append(s)
    print("reachable nodes", reachable)

    # Step 3: Keep only triples with reachable nodes
    new_triples = []
    for t in non_location_triples:
        print("t", t)
        if t[0] in reachable and (isinstance(t[2], str) or t[2] in reachable):
            new_triples.append(t)
    print("new triples", new_triples)
    # Create a new graph
    print("about to create new graph")
    original_graph = penman.Graph(instances, top=top)
    print("original graph", original_graph)

    print("original graph encoding", penman.encode(original_graph))
    new_graph = penman.Graph(new_triples, top=top)
    print("new graph", new_graph)
    try:
        return penman.encode(new_graph)
    except penman.exceptions.LayoutError:
        print("Warning: Could not encode modified graph. Returning original.")
        return amr_string
