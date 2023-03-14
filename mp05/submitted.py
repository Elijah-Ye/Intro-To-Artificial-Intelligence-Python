# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue as q

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    path = []
    start = maze.start
    if(maze[start[0], start[1]] == maze.legend.waypoint):
        path.append(start)
        return path
    
    queue = []
    queue.append(start)
    
    visited = set()
    visited.add(start)
    
    prev = {}

    while queue:
        cell = queue.pop(0)
        if(maze[cell[0], cell[1]] == maze.legend.waypoint):
            path = [cell]
            while path[-1] != maze.start:
                path.append(prev[path[-1]])
            path.reverse()
            return path   
        
        neighbors = maze.neighbors(cell[0],cell[1])
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in queue:
                prev[neighbor] = cell
                queue.append(neighbor)
                visited.add(neighbor) 
    
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    path = []
    start = maze.start
    if(maze[start[0], start[1]] == maze.legend.waypoint):
        path.append(start)
        return path
    
    priority_queue = q.PriorityQueue()
    
    end = maze.waypoints[0]
    
    startA = (manhattan_distance(start, end), start)
    
    
    visited = set()
    visited.add(start)
    
    prev = {}
    priority_queue.put(startA)

    while priority_queue:
        cellA = priority_queue.get()
        cell = cellA[1]
        if(maze[cell[0], cell[1]] == maze.legend.waypoint):
            return path_between(start, cell, prev)
        
        neighbors = maze.neighbors(cell[0],cell[1])
        for neighbor in neighbors:
            if neighbor not in visited:
                prev[neighbor] = cell
                neighborA = (manhattan_distance(neighbor, end) + len(path_between(start, cell, prev)), neighbor)
                priority_queue.put(neighborA)
                visited.add(neighbor) 
    
    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    

    return []

# This function returns the manhattan distance between two maze cells
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def path_between(a, b, prev):
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    return path  


