!pip install haversine
import math
import heapq
import time
import geopandas as gpd
import folium
import pandas as pd
import numpy as np
from IPython.display import display
import folium
from haversine import haversine as hs

# Load the shapefile from a GitHub URL and reproject to EPSG:4326 (lat/lon)
github_url = "https://raw.githubusercontent.com/bneelkamal/ArtificialIntelligence/main/Census_2011/2011_Dist.shp"
talukas = gpd.read_file(github_url)
talukas = talukas.to_crs("EPSG:4326")

# Calculate centroids for each district
talukas["centroid"] = talukas.geometry.centroid
talukas["lon"] = talukas["centroid"].x
talukas["lat"] = talukas["centroid"].y

# Identify neighboring districts using spatial join
neighbors = gpd.sjoin(talukas[["geometry"]], talukas[["geometry"]], predicate="touches", how='left')

graph = {}
for idx1, row1 in talukas.iterrows():
    taluka1 = (row1["DISTRICT"], row1["ST_NM"])
    graph[taluka1] = {}
    neighbor_rows = neighbors[neighbors.index == idx1]
    for _, row2 in neighbor_rows.iterrows():
        if pd.notna(row2["index_right"]):
            taluka2 = (talukas.loc[row2["index_right"], "DISTRICT"],
                       talukas.loc[row2["index_right"], "ST_NM"])  # Node for neighbor
            dist = hs((row1["lat"], row1["lon"]),
                             (talukas.loc[row2["index_right"], "lat"],
                             talukas.loc[row2["index_right"], "lon"]))
            graph[taluka1][taluka2] = dist * 2  # Travel time approximation


# Dijkstra's algorithm for shortest paths (used in heuristic)
def dijkstra(graph, start):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        curr_dist, curr = heapq.heappop(pq)
        if curr_dist > distances[curr]:
            continue
        for neighbor, weight in graph[curr].items():
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# Precompute shortest paths for all districts
shortest_paths = {taluka: dijkstra(graph, taluka) for taluka in graph}

# Heuristic: minimum of maximum times to a common meeting point
def heuristic(state):
    A, B = state
    if A == B:
        return 0
    return min(max(shortest_paths[A][M], shortest_paths[B][M]) for M in graph)

# Generate neighboring states
def get_neighbors(state):
    A, B = state
    A_options = list(graph[A].keys())
    B_options = list(graph[B].keys())

    # Handle the case where both friends are initially in the same location
    if A == B:
        return [((a, b), cost)
                for a in A_options
                for b in B_options
                if a != b  # Ensure they don't move to the same place
                for cost in [max(graph[A].get(a, 0), graph[B].get(b, 0))]]

    # Generate neighbors with the constraint that at least one friend moves
    neighbors = []
    for a in A_options:
        for b in B_options:
            cost = max(graph[A].get(a, 0), graph[B].get(b, 0))  # Get transition cost
            neighbors.append(((a, b), cost))

    # If A's options include staying in place, add moves where only B moves
    if A in graph[A]:
        for b in B_options:
            if b != B:  # Ensure B moves
                cost = graph[B][b]
                neighbors.append(((A, b), cost))

    # If B's options include staying in place, add moves where only A moves
    if B in graph[B]:
        for a in A_options:
            if a != A:  # Ensure A moves
                cost = graph[A][a]
                neighbors.append(((a, b), cost))

    return neighbors

# Transition cost: maximum time between current and next state
def transition_cost(state, next_state):
    A, B = state
    C, D = next_state
    time_me = 0 if A == C else graph[A].get(C, float("inf"))
    time_friend = 0 if B == D else graph[B].get(D, float("inf"))
    return max(time_me, time_friend)

# Reconstruct path from start to goal
def reconstruct_path(came_from, start, goal):
    path = [goal]
    while goal != start:
        goal = came_from[goal]
        path.append(goal)
    return path[::-1]

# Greedy Best-First Search (GBFS)
def greedy_best_first_search(start,heuristic_func):
    start_time = time.time()
    frontier = [(heuristic_func(start), start)]
    came_from = {}
    nodes_generated = 1
    max_space = 1
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current[0] == current[1]:  # Friends meet
            break
        if current in visited:
            continue
        visited.add(current)
        for next_state, _ in get_neighbors(current):
            if next_state not in visited and next_state not in came_from:
                came_from[next_state] = current
                nodes_generated += 1
                heapq.heappush(frontier, (heuristic_func(next_state), next_state))
                max_space = max(max_space, len(frontier))

    # Reconstruct path
    path = reconstruct_path(came_from, start, current)

    # Calculate total cost
    total_cost = sum(transition_cost(path[i], path[i+1]) for i in range(len(path) - 1))
    execution_time = time.time() - start_time

    return path, total_cost, nodes_generated, max_space, execution_time

# A* Search
def a_star_search(start,heuristic_func):
    start_time = time.time()
    g_score = {start: 0}
    f_score = {start: heuristic_func(start)}
    frontier = [(f_score[start], start)]
    came_from = {}
    nodes_generated = 1
    max_space = 1
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current[0] == current[1]:  # Friends meet
            break
        if current in visited:
            continue
        visited.add(current)
        for next_state, step_cost in get_neighbors(current):
            tentative_g = g_score[current] + transition_cost(current, next_state)
            if next_state not in g_score or tentative_g < g_score[next_state]:
                came_from[next_state] = current
                g_score[next_state] = tentative_g
                f_score[next_state] = tentative_g + heuristic_func(next_state)
                if next_state not in visited:
                    heapq.heappush(frontier, (f_score[next_state], next_state))
                    nodes_generated += 1
                    max_space = max(max_space, len(frontier))

    # Reconstruct path
    path = reconstruct_path(came_from, start, current)

    # Total cost is g_score of the meeting point
    total_cost = g_score[current]
    execution_time = time.time() - start_time

    return path, total_cost, nodes_generated, max_space, execution_time


def calculate_distance(district1, district2):
    # Get coordinates using both district and state names
    coord1 = talukas.loc[(talukas['DISTRICT'] == district1[0]) &
                         (talukas['ST_NM'] == district1[1]),
                         ['lat', 'lon']].iloc[0]
    coord2 = talukas.loc[(talukas['DISTRICT'] == district2[0]) &
                         (talukas['ST_NM'] == district2[1]),
                         ['lat', 'lon']].iloc[0]

    # Calculate distance using haversine function
    return hs((coord1['lat'], coord1['lon']), (coord2['lat'], coord2['lon']))

def add_path_with_distance(m, coords, color, label, path_districts):
    total_distance = 0
    for i in range(len(coords) - 1):
        dist = calculate_distance(path_districts[i], path_districts[i + 1])
        total_distance += dist
        segment_popup = f"{label} Segment {i + 1}: {dist:.2f} km<br>Total Distance: {total_distance:.2f} km"
        folium.PolyLine(
            locations=[coords[i], coords[i + 1]],
            color=color,
            weight=2.5,
            opacity=1,
            popup=segment_popup
        ).add_to(m)

def create_folium_map(path, title):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')

    # Helper function to get coordinates
    def get_coords(district, state):
        filtered = talukas[(talukas['DISTRICT'] == district) & (talukas['ST_NM'] == state)]
        if filtered.empty:
            print(f"Warning: No match found for {district}, {state}")
            return None
        return tuple(filtered[['lat', 'lon']].iloc[0])

    # Extract coordinates for both friends
    friend1_coords = [get_coords(step[0][0], step[0][1]) for step in path]
    friend2_coords = [get_coords(step[1][0], step[1][1]) for step in path]

    # Filter out None values (invalid coordinates)
    friend1_coords = [c for c in friend1_coords if c is not None]
    friend2_coords = [c for c in friend2_coords if c is not None]

    if not friend1_coords or not friend2_coords:
        print("Error: Insufficient coordinates to generate map.")
        return None

# Add start markers using district and state
    folium.Marker(
        location=friend1_coords[0],
        popup=f"Start: Friend 1 ({path[0][0][0]}, {path[0][0][1]})",  # District, State
        icon=folium.Icon(color='blue')
    ).add_to(m)
    folium.Marker(
        location=friend2_coords[0],
        popup=f"Start: Friend 2 ({path[0][1][0]}, {path[0][1][1]})",  # District, State
        icon=folium.Icon(color='red')
    ).add_to(m)

    # Add meeting point marker using district and state
    meeting_point = friend1_coords[-1]  # Assuming they meet at the end
    folium.Marker(
        location=meeting_point,
        popup=f"Meeting Point: ({path[-1][0][0]}, {path[-1][0][1]})",  # District, State
        icon=folium.Icon(color='green', icon='star')
    ).add_to(m)

    # Add detailed markers for intermediate points using district and state
    for i, coord in enumerate(friend1_coords[1:-1], 1):  # Skip start and end
        district, state = path[i][0]
        folium.Marker(
            location=coord,
            popup=f"Friend 1: {district}, {state}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    for i, coord in enumerate(friend2_coords[1:-1], 1):  # Skip start and end
        district, state = path[i][1]
        folium.Marker(
            location=coord,
            popup=f"Friend 2: {district}, {state}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # Add paths (using add_path_with_distance function)
    add_path_with_distance(m, friend1_coords, 'blue', 'Friend 1', [step[0] for step in path])
    add_path_with_distance(m, friend2_coords, 'red', 'Friend 2', [step[1] for step in path])

    return m

def euclidean_heuristic(state):
    A, B = state
    # Get district A's coordinates
    rowA = talukas[(talukas['DISTRICT'] == A[0]) & (talukas['ST_NM'] == A[1])]
    # Get district B's coordinates
    rowB = talukas[(talukas['DISTRICT'] == B[0]) & (talukas['ST_NM'] == B[1])]
    if rowA.empty or rowB.empty:
        return float("inf")
    coord_A = (rowA.iloc[0]['lat'], rowA.iloc[0]['lon'])
    coord_B = (rowB.iloc[0]['lat'], rowB.iloc[0]['lon'])
    # Use the haversine function (hs imported from haversine package)
    return hs(coord_A, coord_B)

# Define starting points
S_me = "Pune","Maharashtra"
S_friend = "Bangalore","Karnataka"
initial_state = (S_me, S_friend)

# Run the searches
gbfs_result = greedy_best_first_search(initial_state, heuristic)
astar_result = a_star_search(initial_state , heuristic)

# Print the results
print("Greedy Best-First Search: heuristic")
print(f"Path: {gbfs_result[0]}")
print(f"Total Cost: {gbfs_result[1]}")
print(f"Nodes Generated: {gbfs_result[2]}")
print(f"Max Space: {gbfs_result[3]}")
print(f"Execution Time: {gbfs_result[4]:.4f}s\n")

# Display maps inline
print("Greedy Best-First Search Map:")
gbfs_map = create_folium_map(gbfs_result[0], "Greedy Best-First Search")
display(gbfs_map)

print("A* Search: heuristic")
print(f"Path: {astar_result[0]}")
print(f"Total Cost: {astar_result[1]}")
print(f"Nodes Generated: {astar_result[2]}")
print(f"Max Space: {astar_result[3]}")
print(f"Execution Time: {astar_result[4]:.4f}s")

print("A* Search Map:")
astar_map = create_folium_map(astar_result[0], "A* Search")
display(astar_map)

# Run the searches
gbfs_result_euclidean = greedy_best_first_search(initial_state, euclidean_heuristic)

astar_result_euclidean = a_star_search(initial_state , euclidean_heuristic)

# Print the results
print("Greedy Best-First Search: euclidean_heuristic")
print(f"Path: {gbfs_result_euclidean[0]}")
print(f"Total Cost: {gbfs_result_euclidean[1]}")
print(f"Nodes Generated: {gbfs_result_euclidean[2]}")
print(f"Max Space: {gbfs_result_euclidean[3]}")
print(f"Execution Time: {gbfs_result_euclidean[4]:.4f}s\n")

# Display maps inline
print("Greedy Best-First Search Map: euclidean_heuristic")
gbfs_map_euclidean = create_folium_map(gbfs_result_euclidean[0], "Greedy Best-First Search")
display(gbfs_map_euclidean)


print("A* Search: euclidean_heuristic")
print(f"Path: {astar_result_euclidean[0]}")
print(f"Total Cost: {astar_result_euclidean[1]}")
print(f"Nodes Generated: {astar_result_euclidean[2]}")
print(f"Max Space: {astar_result_euclidean[3]}")
print(f"Execution Time: {astar_result_euclidean[4]:.4f}s")

print("A* Search Map: euclidean_heuristic")
astar_map_euclidean = create_folium_map(astar_result_euclidean[0], "A* Search")
display(astar_map_euclidean)
