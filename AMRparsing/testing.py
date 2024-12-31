import penman

roles_to_remove = [':ARG2']

amr = '(b / believe-01 :ARG0 (g / girl) :ARG1 (s / sit-01 :ARG1 (c / cat) :ARG2 (m / mat)))'
tree = penman.parse(amr)
location_paths = []
edges = tree.walk()
l = list(edges)
i = 0
for path, branch in tree.walk():
  i+=1
  role, target = branch
  if role in roles_to_remove:
    location_paths.insert(0,path)

top = tree.nodes()[0]
for path in location_paths:
  node = top
  for index in path[:-1]:
    temp = node[1]
    temp2 = node[1][index]
    node = node[1][index][1]
  node[1].pop(path[-1])

new_tree = penman.Tree(top)
new_amr = penman.format(new_tree)

print(new_amr)