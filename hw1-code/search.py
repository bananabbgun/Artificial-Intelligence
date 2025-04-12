# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import maze
from collections import deque
from queue import PriorityQueue
import itertools
import math



def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def manhattan(a, b):
    x1, y1 = a
    x2, y2 = b
    return abs(x1 - x2) + abs(y1 - y2)

def bfs(maze):
    start = maze.getStart()
    queue = deque([(start, [start])])
    visited = set([start]) 
    
    while queue:
        cur, path = queue.popleft()
        
        if cur in maze.getObjectives():
            return path  
        
        for neighbor in maze.getNeighbors(*cur):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


def astar(maze):
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    pq = PriorityQueue()
    pq.put((0, start))  
    comefrom = {start: None}
    cost = {start: 0}
    while not pq.empty():
        cur = pq.get()[1]

        if cur == goal:
            break

        for next in maze.getNeighbors(*cur):
            new_cost = cost[cur] + 1  
            if next not in cost or new_cost < cost[next]:
                cost[next] = new_cost
                priority = new_cost + manhattan(next, goal)
                pq.put((priority, next))
                comefrom[next] = cur
    cur = goal
    path = []
    while cur != start:
        path.append(cur)
        cur = comefrom[cur]
    path.append(start) 
    path.reverse()
    
    return path
    



mst_tree = {}
dis = {}

def bfs_path(maze, start, goal, typ = 0):
    global dis
    if (start, goal) in dis:
        return dis[(start, goal)]

    if typ == 0 : 
        bfs_path(maze, goal, start,1)
    queue = deque([(start, 0)])
    visited = set([start]) 
    ct = 0
    
    while queue:
        cur, cost = queue.popleft()
        dis[(start, cur)] = dis[(cur, start)] = cost
        if cur == goal and ct == 0:
            ct = 1
            ans = cost  
        
        for neighbor in maze.getNeighbors(*cur):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, cost + 1))
    return ans
     


def calculate_mst(maze, remaining_dots):
    if tuple(remaining_dots) in mst_tree:
        return mst_tree[tuple(remaining_dots)]
    pq = PriorityQueue()
    start = remaining_dots[0]
    visited = set()
    visited.add(start)
    mst_weight = 0
    edge_weights = {dot: float('inf') for dot in remaining_dots}
    edge_weights[start] = 0

    for dot in remaining_dots:
        if dot != start:
            if (start, dot) not in dis:
                weight = bfs_path(maze, start, dot)
                dis[(start, dot)] = weight
                dis[(dot, start)] = weight
            pq.put((dis[(start, dot)], start, dot))

    while len(visited) < len(remaining_dots):
        weight, u, v = pq.get()
        if v in visited:
            continue

        visited.add(v)
        mst_weight += weight

        for next_dot in remaining_dots:
            if next_dot not in visited:
                if (v, next_dot) not in dis:
                    dis[(v, next_dot)] = dis[(next_dot, v)] = bfs_path(maze, v, next_dot)
                new_weight = dis[(v, next_dot)]
                
                if new_weight < edge_weights[next_dot]:
                    pq.put((new_weight, v, next_dot))
                    edge_weights[next_dot] = new_weight

    mst_tree[tuple(remaining_dots)] = mst_weight
    return mst_weight

near = {}
def nearest(maze, cur_pos, remaining_dots):
    if (cur_pos, tuple(remaining_dots)) not in near:
        near[(cur_pos, tuple(remaining_dots))] = min([bfs_path(maze, cur_pos, dot) for dot in remaining_dots])
    return near[(cur_pos, tuple(remaining_dots))]

def heuristic(maze, cur_pos, remaining_dots):
    if len(remaining_dots) == 0:
        return 0
    mst = calculate_mst(maze, remaining_dots)
    return mst + nearest(maze, cur_pos, remaining_dots)

def reconstruct_path(comefrom, start, goal):
    cur = goal
    path = []
    while cur != start:
        path.append(cur[0])
        cur = comefrom[cur]
    path.append(start[0]) 
    path.reverse()
    return path
def astar_corner(maze):
    start = maze.getStart()
    dots = maze.getObjectives()
    pq = PriorityQueue()
    pq.put((heuristic(maze, start, dots), (start, dots)))
    comefrom = {(start, tuple(dots)): None}
    cost = {(start, tuple(dots)): 0}
    cur_pos = start


    while not pq.empty():
        _, (cur_pos, remaining_dots) = pq.get()
        if len(remaining_dots) == 0:
            return reconstruct_path(comefrom, (start, tuple(dots)), (cur_pos, tuple(remaining_dots)))

        for next_pos in maze.getNeighbors(cur_pos[0], cur_pos[1]):
            
            new_remain = list(dot for dot in remaining_dots if dot != next_pos)
            new_cost = cost[(cur_pos, tuple(remaining_dots))] + 1
            if (next_pos, tuple(new_remain)) not in cost or new_cost < cost[(next_pos, tuple(new_remain))]:
                cost[(next_pos, tuple(new_remain))] = new_cost
                priority = new_cost + heuristic(maze, next_pos, new_remain)
                pq.put((priority, (next_pos, new_remain)))
                comefrom[(next_pos, tuple(new_remain))] = (cur_pos, tuple(remaining_dots))
    return []
def astar_multi(maze):
    start = maze.getStart()
    dots = maze.getObjectives()
    pq = PriorityQueue()
    pq.put((heuristic(maze, start, dots), (start, dots)))
    comefrom = {(start, tuple(dots)): None}
    cost = {(start, tuple(dots)): 0}
    cur_pos = start


    while not pq.empty():
        _, (cur_pos, remaining_dots) = pq.get()
        if len(remaining_dots) == 0:
            return reconstruct_path(comefrom, (start, tuple(dots)), (cur_pos, tuple(remaining_dots)))

        for next_pos in maze.getNeighbors(cur_pos[0], cur_pos[1]):
            
            new_remain = list(dot for dot in remaining_dots if dot != next_pos)
            new_cost = cost[(cur_pos, tuple(remaining_dots))] + 1
            if (next_pos, tuple(new_remain)) not in cost or new_cost < cost[(next_pos, tuple(new_remain))]:
                cost[(next_pos, tuple(new_remain))] = new_cost
                priority = new_cost + heuristic(maze, next_pos, new_remain)
                pq.put((priority, (next_pos, new_remain)))
                comefrom[(next_pos, tuple(new_remain))] = (cur_pos, tuple(remaining_dots))

    return []




def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    objectives = maze.getObjectives()
    path = []
    cur_pos = start
    
    while objectives:
        nearest_dot = min(objectives, key=lambda dot: manhattan(cur_pos, dot))
        
        temp_path = simple_astar(maze, cur_pos, nearest_dot)
        
        if path and temp_path:
            temp_path = temp_path[1:]
        
        path.extend(temp_path)  
        cur_pos = nearest_dot  
        objectives.remove(nearest_dot)  
        
    return path

def simple_astar(maze, start, goal):
    pq = PriorityQueue()
    pq.put((0 + manhattan(start, goal), 0, start, [start]))
    visited = set()
    
    while not pq.empty():
        _, g_score, current, path = pq.get()
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            return path
        
        for neighbor in maze.getNeighbors(current[0], current[1]):
            if neighbor in visited:
                continue
            new_g = g_score + 1
            new_f = new_g + manhattan(neighbor, goal)
            pq.put((new_f, new_g, neighbor, path + [neighbor]))
    
    return []
