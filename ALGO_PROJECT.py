from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import sys


def readingData(filenameArg):
    filename = 'D:\\FAST- Nuces\\Semester 05\\Algorithms\\benchmark\\'+filenameArg
    with open(filename) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)

    count = 0
    list1 = []
    Node = []

    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list1.append(listli)

    v = int(list1[1][0])
    # Adjacentcy Matrix
    adjacent = [[0] * v for _ in range(v)]

    # for all nodes 0 to n nodes
    for i in range(0, v):
        ps = (float(list1[2 + i][1]), float(list1[2 + i][2]))
        Node.append(ps)

    # skipping nodes + 2(netsin and num of nodes)
    for i in range(v + 2, len(list1) - 1):
        f = int(list1[i][0])                        #from vertex

        for j in range(1, len(list1[i]), 4):
            t = int(list1[i][j])                    # to vertex
            w = float(list1[i][j + 2])              # weight

            edge = (int(f), int(t), float(w))
            if adjacent[f][t] > w or adjacent[f][t] == 0:
                adjacent[f][t] = w

    source = int(list1[len(list1)-1][0])
    return source,adjacent,v,Node


def changeUndirect(graph):
    for i in range(0,len(graph)):
        for j in range(0,len(graph[i])):
          if(graph[i][j]!=0 and graph[i][j]!=graph[j][i]):
              min = graph[i][j]
              if(graph[j][i]!=0 and graph[j][i]<min):
                  min = graph[j][i]
              graph[i][j]=min
              graph[j][i]=min
    return graph;


def printUnDirectedGraph(adjMat,pos,v):
    g = nx.Graph()

    for i in range(0,v):
        po = (pos[i][0],pos[i][1])
        g.add_node(i,pos=po)

    for i in range(0,v):
        for j in range(0,v):
            if(adjMat[i][j]!=0):
                weight = adjMat[i][j]/1000000
                g.add_edge(i,j,weight=weight)
    weight = nx.get_edge_attributes(g, 'weight')
    pos1 = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos1, edge_labels=weight)
    nx.draw(g, pos1, with_labels=1, font_color='yellow')
    plt.show()


def printDirectedGraph(adjMat,pos):
    g = nx.DiGraph()

    for i in range(0,len(adjMat)):
        po = pos[i]
        g.add_node(i,pos=po)

    for i in range(0,len(adjMat)):
        for j in range(0,len(adjMat[i])):
            if(adjMat[i][j]!=0):
                weight = adjMat[i][j]/1000000
                g.add_edge(i,j,weight = weight)
    weight = nx.get_edge_attributes(g, 'weight')
    pos1 = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos1, edge_labels=weight)
    nx.draw(g, pos1, with_labels=1, font_color='yellow')
    plt.show()

def printInputGraph():
    print(InputFileClicked.get())
    with open('D:\\FAST- Nuces\\Semester 05\\Algorithms\\benchmark\\'+InputFileClicked.get()) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)
    g = nx.DiGraph()
    count = 0
    list = []
    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list.append(listli)
    v = int(list.__getitem__(1).__getitem__(0))
    print(v)
    for i in range(0, v):
        ps = (float(list.__getitem__(2 + i).__getitem__(1)), float(list.__getitem__(2 + i).__getitem__(2)))
        g.add_node(i, pos=ps)
    i = 0
    j = 0
    for i in range(v + 2, len(list) - 1):
        f = int(list.__getitem__(i).__getitem__(0))
        for j in range(1, len(list.__getitem__(i)), 4):
            t = int(list.__getitem__(i).__getitem__(j))
            w = float(list.__getitem__(i).__getitem__(j + 2))
            w=w/1000000;
            g.add_edge(f, t, weight=w)
    weight = nx.get_edge_attributes(g, 'weight')
    pos = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight)
    nx.draw(g, pos, with_labels=1, font_color='yellow')
    plt.show()



#   ---------- PRIMS ALGORITHMS

def findMinVertex(visited, weights,V):
    index = -1;
    minW = sys.maxsize;
    for i in range(V):
        if (visited[i] == False and weights[i] < minW):
            minW = weights[i];
            index = i;
    return index;

def PrimsAlgo(graph,V,S):
    visited = [True] * V;
    weights = [0] * V;
    parent = [0] * V;

    for i in range(V):
        visited[i] = False;
        weights[i] = sys.maxsize;

    weights[S] = 0;
    parent[S] = -1;

    for i in range(V - 1):
        minVertex = findMinVertex(visited, weights,V);
        visited[minVertex] = True;
        for j in range(V):
            if (graph[minVertex][j] != 0 and visited[j] == False):
                if (graph[minVertex][j] < weights[j]):
                    weights[j] = graph[minVertex][j];
                    parent[j] = minVertex;
    for i in range(V):
            for j in range(V):
                graph[i][j]=0
                if(parent[j]==i):
                    graph[i][j]=weights[j]
    return graph


#-----------------Kruskal Alorithm

def kunion(i, j,parent):
    a = find(parent,i)
    b = find(parent,j)
    parent[a] = b


def kruskalMST(cost,V):
    G = [[0] * V for _ in range(V)]
    parent=[0]*V

    # Initialize sets of disjoint sets
    for i in range(V):
        parent[i] = i

    # Include minimum weight edges one by one
    edge_count = 0
    while edge_count < V - 1:
        min = sys.maxsize
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(parent,i) != find(parent,j) and cost[i][j] < min and cost[i][j]!=0:
                    min = cost[i][j]
                    a = i
                    b = j
        kunion(a, b,parent)
        print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, min))
        G[a][b]=min
        G[b][a] = min
        edge_count += 1

    return G



#--------- Bellmen ford
def BellmenFord(graph, V, src):
    dist = [sys.maxsize] * V
    dist[src] = 0
    parent = [-1]*V

    for q in range(V - 1):
        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                x = i
                y = j
                w = graph[i][j]

                if dist[x] + w < dist[y]:
                    dist[y] = dist[x] + w
                    parent[y]=x

    for i in range(V):
        for j in range(V):
            if graph[i][j] == 0:
                continue
            x = i
            y = j
            w = graph[i][j]
            if dist[x] != sys.maxsize and dist[x] + w < dist[y]:
                return None

    for i in range(V):
        for j in range(V):
            graph[i][j]=0

    for i in range(V):
        if(parent[i]!=-1):
            graph[parent[i]][i] = dist[i]
    return graph



# - -------- Dijsktra
def minDistance(V, dist, sptSet):
    min = sys.maxsize
    min_index = -1

    # Search not nearest vertex not in the
    # shortest path tree
    for u in range(V):
        if dist[u] < min and sptSet[u] == False:
            min = dist[u]
            min_index = u

    return min_index


def dijkstra(G, V, src):
    dist = [sys.maxsize] * V
    sptSet = [False] * V
    parent = [-1] * V
    dist[src] = 0
    for cout in range(V):

        x = minDistance(V, dist, sptSet)
        sptSet[x] = True


        for y in range(0, V):

            if (G[x][y] > 0 and sptSet[y] == False) and (dist[y] > dist[x] + G[x][y]):
                dist[y] = dist[x] + G[x][y]
                parent[y] = x
    for i in range(V):
        for j in range(V):
            G[i][j] = 0
    for i in range(V):
        if (parent[i] != -1):
            G[parent[i]][i] = dist[i]
    return G




def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def boruvka(graph, V):
    parent = []
    rank = []
    G  = [[0] * V for _ in range(V)]

    cheapest = []

    numTrees = V
    MSTweight = 0

    for node in range(V):
        parent.append(node)
        rank.append(0)
        cheapest = [-1] * V


    while numTrees > 1:

        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                w = graph[i][j]
                set1 = find(parent, i)
                set2 = find(parent, j)

                if set1 != set2:

                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [i, j, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [i, j, w]
        for node in range(V):


            if cheapest[node] != -1:
                u, v, w = cheapest[node]
                set1 = find(parent, u)
                set2 = find(parent, v)

                if set1 != set2:
                    MSTweight += w
                    union(parent, rank, set1, set2)
                    G[u][v] = w
                    print("Edge %d-%d with weight %d included in MST" % (u, v, w))
                    numTrees = numTrees - 1


        cheapest = [-1] * V
    return G



#   floyd warshal


def floydWarshal(graph):
    dist = graph
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(i==j):
                dist[i][j] = 0
            elif(graph[i][j]==0):
                dist[i][j] = sys.maxsize
            else:
                dist[i][j]=graph[i][j]
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j],dist[i][k] + dist[k][j])
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(dist[i][j]==sys.maxsize):
                graph[i][j] = 0
            else:
                graph[i][j] = dist[i][j]
    return graph

# Clustering


def Clustering_Coefficient(adjM,p,v):
    g = nx.Graph()

    for i in range(0, v):
        po = (p[i][0], p[i][1])
        g.add_node(i, pos=po)

    for i in range(0, v):
        for j in range(0, v):
            if (adjM[i][j] != 0):
                weight = adjM[i][j]
                g.add_edge(i, j, weight=weight)
    print(nx.average_clustering(g))




#when user will click show result button
def showResultFunc():
    s, adjM, v, p = readingData(InputFileClicked.get())
    if Algoclicked.get() == algorithms[0]:
        adjM = changeUndirect(adjM)
        adjM = PrimsAlgo(adjM,v,s)
        printUnDirectedGraph(adjM,p,v)
    elif Algoclicked.get() == algorithms[3]:
        adjM = BellmenFord(adjM, v, s)
        printDirectedGraph(adjM, p)
    elif Algoclicked.get()==algorithms[2]:
        adjM = dijkstra(adjM,v,s)
        printDirectedGraph(adjM,p)
    elif Algoclicked.get()==algorithms[4]:
        adjM = floydWarshal(adjM)
        printDirectedGraph(adjM, p)
    elif Algoclicked.get()==algorithms[5]:
        adjM = changeUndirect(adjM)
        Clustering_Coefficient(adjM,p,v)
    elif Algoclicked.get()==algorithms[6]:
        adjM = boruvka(adjM,v)
        printDirectedGraph(adjM,p)
    elif Algoclicked.get()==algorithms[1]:
        adjM = changeUndirect(adjM)
        print(adjM)
        adjM = kruskalMST(adjM,v)
        printUnDirectedGraph(adjM,p,v)


backGroundColor = 'black'
textColor = 'white'
root = Tk()
root.geometry("800x400")
root.configure(bg=backGroundColor)
horizontal_layout_dropDowns = Frame(root)
inputFiles = [ "input10.txt", "input20.txt", "input30.txt", "input40.txt", "input50.txt", "input60.txt", "input70.txt", "input80.txt", "input90.txt", "input100.txt"]
algorithms = [ "Prims", "Kruskal", "Dijkstra", "Bellmen Ford", "Floyd Warshal", "Clustering Coefficient", "Borůvka's algorithm"]
Algoclicked = StringVar()
Algoclicked.set(algorithms[0])
AlogDropDown = OptionMenu(horizontal_layout_dropDowns, Algoclicked, *algorithms)
AlogDropDown.config(bg='black', fg='white')
AlogDropDown.grid(row=5, column=1)
InputFileClicked = StringVar()
InputFileClicked.set(inputFiles[0])
InputFileDropDown = OptionMenu(horizontal_layout_dropDowns, InputFileClicked, *inputFiles)
InputFileDropDown.config(bg='black', fg='white')
# InputFileDropDown["borderwidth"]=0
InputFileDropDown.grid(row=5, column=2, pady=5, padx=5)
horizontal_layout_dropDowns.configure(bg=backGroundColor, pady=5)
horizontal_layout_dropDowns.pack()

horizontal_layout_button = Frame(root)
ShowInput = Button(horizontal_layout_button, text="Show input graph", bg='black', fg='white', padx=5, pady=5, command=printInputGraph)
ShowInput.grid(row=10, column=2, padx=10, pady=20)

ShowResult = Button(horizontal_layout_button, text="Show result graph", bg='black', fg='white', padx=5, pady=5,command=showResultFunc)
ShowResult.grid(row=10, column=5, padx=10, pady=20)
horizontal_layout_button.configure(bg=backGroundColor)
horizontal_layout_button.pack()
root.mainloop()