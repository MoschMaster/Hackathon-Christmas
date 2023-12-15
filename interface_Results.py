import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Custom CSS for styling
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math



# Read the file
file_path = 'Datasets/2_Delft.txt'

# Read the general parameters line
with open(file_path, 'r') as file:
    general_params = file.readline()
    general_paramslist = general_params.split(" ")


horizontal_streets = int(general_paramslist[0])
vertical_street = int(general_paramslist[1])
no_sledges = int(general_paramslist[2])
no_presents = int(general_paramslist[3])
bonus_per_ride = int(general_paramslist[4])
no_steps = int(general_paramslist[5]) 



class Present:
    def __init__(self, a, b, x, y, s, f,n):
        self.starthor = a
        self.startver = b
        self.finhor = x
        self.finver = y
        self.devtime = s
        self.fintime = f
        self.nr = n

class Sledge:
    def __init__(self):
        self.no_pres_deliv = 0
        self.no_steps_taken = 0
        self.pres_deliv = []
        self.current_hor = 0
        self.current_ver = 0

    def add_present(self, p):
        self.no_pres_deliv += 1
        self.no_steps_taken += (abs(self.current_hor - p.starthor) + abs(self.current_ver - p.startver) + 
                                abs(p.starthor - p.finhor) + abs(p.startver - p.finver))
        self.pres_deliv.append(p)
        self.current_hor, self.current_ver = p.finhor, p.finver


def nearest_neighbor_route(df, sledges, max_steps):
    unvisited = [Present(row['Start_Horizontal'], row['Start_Vertical'], row['Finish_Horizontal'], 
                         row['Finish_Vertical'], row['Earliest_Start'], row['Latest_Finish'],index) for index, row in df.iterrows()]

    for sledge in sledges:
        steps_taken = 0  # Initialize steps taken for each sledge
        while unvisited:
            nearest_present, nearest_idx = None, None
            min_distance = float('inf')

            # Find the nearest unvisited present
            for idx, present in enumerate(unvisited):
                distance = abs(sledge.current_hor - present.starthor) + abs(sledge.current_ver - present.startver)
                if distance < min_distance:
                    min_distance = distance
                    nearest_present = present
                    nearest_idx = idx

            if nearest_present:
                # Calculate the time to reach the present's starting point
                time_to_start = steps_taken + min_distance
                if time_to_start < nearest_present.devtime:
                    # If the sledge arrives before devtime, wait until devtime
                    steps_taken += nearest_present.devtime - time_to_start

                # Check if the present can be delivered within its time constraints and maximum steps
                present_distance = abs(nearest_present.starthor - nearest_present.finhor) + abs(nearest_present.startver - nearest_present.finver)
                if steps_taken + min_distance + present_distance <= max_steps and \
                   sledge.no_pres_deliv < no_presents and \
                   sledge.no_pres_deliv < no_sledges and \
                   nearest_present.devtime >= steps_taken and \
                   nearest_present.fintime >= steps_taken + min_distance + present_distance:
                    sledge.add_present(nearest_present)
                    del unvisited[nearest_idx]

                    # Update steps taken for this sledge
                    steps_taken += min_distance + present_distance

                else:
                    # Present doesn't meet constraints or exceeds max steps, move to the next present
                    nearest_present = None

            if nearest_present is None:
                # If no present was selected or steps exceeded, break the loop for this sledge
                break

df = pd.read_csv(file_path, sep=' ', skiprows=0, header=None)
df.columns = ['Start_Horizontal', 'Start_Vertical', 'Finish_Horizontal', 'Finish_Vertical', 'Earliest_Start', 'Latest_Finish']



# Create sledges
sledges = [Sledge() for _ in range(no_sledges)]

# Run the nearest neighbor algorithm
nearest_neighbor_route(df, sledges, no_steps)

# Print the routes for each sledge
# for i, sledge in enumerate(sledges):
#     print(f"Sledge {i+1}:")
#     for present in sledge.pres_deliv:
#         print(f"present {present.nr}  Deliver from ({present.starthor}, {present.startver}) to ({present.finhor}, {present.finver})")


def plot_sledge_route(sledge, sledge_number):
    G = nx.DiGraph()
    node_positions = {}  # Dictionary to hold node positions
    prev_point = (0, 0)  # Assuming the sledge starts at (0, 0)
    node_colors = ["green"]  # Starting node color
    G.add_node((0,0))
    node_positions[(0, 0)] = (0, 0)


    for present in sledge.pres_deliv:
        start_point = (present.starthor, present.startver)
        finish_point = (present.finhor, present.finver)

        # Add edges and nodes to the graph
        G.add_node(start_point, color='blue')  # Starting point of the delivery
        G.add_node(finish_point, color='red')  # Ending point of the delivery
        G.add_edge(prev_point, start_point)
        G.add_edge(start_point, finish_point)

        # Assign node positions
        node_positions[start_point] = start_point
        node_positions[finish_point] = finish_point

        # Update the previous point and node colors
        prev_point = finish_point
        node_colors.append('blue')
        node_colors.append('red')

    # Plot the graph
    nx.draw(G, pos=node_positions, node_color=node_colors, with_labels=False, node_size=10, arrows=True)
    plt.title(f"Route for Sledge {sledge_number}")
    plt.show()

# for i, sledge in enumerate(sledges):
#     plot_sledge_route(sledge, i + 1)

def score_routes(sledges, no_steps, bp):
    total_score = 0

    for sledge in sledges:
        # Calculate points for saved steps
        steps_score = no_steps - sledge.no_steps_taken
        total_score += steps_score

        # Calculate points for delivered and timely picked presents
        for present in sledge.pres_deliv:
            if present.devtime == sledge.no_steps_taken - (abs(sledge.current_hor - present.starthor) + abs(sledge.current_ver - present.startver)):
                total_score += bp

        # Subtract points for undelivered presents
        undelivered_presents = no_presents - sledge.no_pres_deliv
        total_score -= 100 * undelivered_presents

    return total_score

# Example usage
score = score_routes(sledges, no_steps, bonus_per_ride)
print(f"Total Score: {score}")

def recalculate_steps(sledge, max_steps=None):
    total_steps = 0
    current_pos = (0, 0)  # Assuming the sledge starts at (0, 0)

    for present in sledge.pres_deliv:
        total_steps += (abs(current_pos[0] - present.starthor) + abs(current_pos[1] - present.startver) +
                        abs(present.starthor - present.finhor) + abs(present.startver - present.finver))
        current_pos = (present.finhor, present.finver)

    if max_steps is not None and total_steps > max_steps:
        return False

    sledge.no_steps_taken = total_steps
    return True

def add_present(sledge, present, max_steps):
    # Determine a random position to insert the present
    insert_index = random.randint(0, len(sledge.pres_deliv))

    # Temporarily add the present to check if it exceeds max steps
    sledge.pres_deliv.insert(insert_index, present)
    sledge.no_pres_deliv += 1

    # Check if the new route exceeds max steps
    if not recalculate_steps(sledge, max_steps):
        # If it exceeds, undo the addition
        sledge.pres_deliv.remove(present)
        sledge.no_pres_deliv -= 1
        return False
    return True

def remove_present(sledge):
    if not sledge.pres_deliv:
        return False

    # Remove a random present from the sledge
    removed_index = random.randint(0, len(sledge.pres_deliv) - 1)
    removed_present = sledge.pres_deliv.pop(removed_index)
    sledge.no_pres_deliv -= 1

    # Recalculate the total steps taken
    recalculate_steps(sledge)
    return True

def modify_solution(sledges, unvisited, max_steps):
    # Choose a random sledge to modify
    sledge_to_modify = random.choice(sledges)

    # Randomly decide to add or remove a present
    if random.choice([True, False]) and unvisited:
        # Try to add a present
        present_to_add = random.choice(unvisited)
        if add_present(sledge_to_modify, present_to_add, max_steps):
            unvisited.remove(present_to_add)
    else:
        # Try to remove a present and add it back to unvisited
        if remove_present(sledge_to_modify):
            # Add the removed present back to unvisited
            present_to_add = sledge_to_modify.pres_deliv[-1]  # Assuming the last present was removed
            unvisited.append(present_to_add)

    return sledges


def simulated_annealing(sledges, no_steps, bp, initial_temp, cooling_rate, no_iterations):
    current_solution = sledges.copy()
    best_solution = sledges.copy()
    current_score = score_routes(current_solution, no_steps, bp)
    best_score = current_score
    temperature = initial_temp

    # Create a list of unvisited presents (initially, all presents are unvisited)
    unvisited = [Present(row['Start_Horizontal'], row['Start_Vertical'], row['Finish_Horizontal'], 
                         row['Finish_Vertical'], row['Earliest_Start'], row['Latest_Finish'], index) 
                 for index, row in df.iterrows()]

    for i in range(no_iterations):
        new_solution = modify_solution(current_solution, unvisited, no_steps)
        new_score = score_routes(new_solution, no_steps, bp)

        if new_score > current_score or random.random() < math.exp((new_score - current_score) / temperature):
            current_solution = new_solution
            current_score = new_score

            if new_score > best_score:
                best_solution = new_solution
                best_score = new_score

        temperature *= cooling_rate

    return best_solution
# Example usage
#optimized_sledges = simulated_annealing(sledges, no_steps, bonus_per_ride, initial_temp=10000, cooling_rate=0.99, no_iterations=100000)

score = score_routes(sledges, no_steps, bonus_per_ride)
print(f"Total Score: {score}")







# Function to read the submission file
def read_submission_file(file_path):
    with open(file_path, 'r') as file:
        sled_routes = {}
        for index, line in enumerate(file):
            parts = line.split()
            sled_routes[f'Sled {index + 1}'] = [int(x) for x in parts[1:]] # Assuming delivery numbers are integers
        return sled_routes

# Function to plot the route (placeholder for actual plotting logic)
def plot_sledge_route(sledge_deliveries, df):
    """
    Plot the route for the selected sledge based on delivery numbers.
    
    :param sledge_deliveries: List of delivery numbers for the selected sledge.
    :param df: DataFrame containing the delivery data.
    """
    plt.figure(figsize=(10, 6))
    prev_point = (0, 0)  # Starting at the depot

    for delivery_num in sledge_deliveries:
        delivery_data = df.loc[delivery_num - 1]  # Adjust for zero-based index
        start_point = (delivery_data['Start_Horizontal'], delivery_data['Start_Vertical'])
        finish_point = (delivery_data['Finish_Horizontal'], delivery_data['Finish_Vertical'])

        # Plot route from previous point to start point, then to finish point
        plt.plot([prev_point[0], start_point[0]], [prev_point[1], start_point[1]], 'bo-')
        plt.plot([start_point[0], finish_point[0]], [start_point[1], finish_point[1]], 'ro-')

        prev_point = finish_point  # Update previous point

    plt.title("Sledge Delivery Route")
    plt.xlabel("Horizontal Coordinate")
    plt.ylabel("Vertical Coordinate")
    plt.grid(True)
    st.pyplot(plt)


# Function to perform and show exploratory analysis
def exploratory_analysis(file_path):
    # Read the data
    data = pd.read_csv(file_path, sep=' ', header=None)
    st.write("## Data Preview")
    st.dataframe(data.head())

    st.write("## Basic Statistics")
    st.table(data.describe())

# Main Streamlit app
def main():
    st.title('🎅 Santa Delivery Route Viewer')

    st.sidebar.header("Options")
    app_mode = st.sidebar.selectbox("Choose the mode", ["Exploratory Analysis", "View Routes"])

    if app_mode == "Exploratory Analysis":
        # ... existing code for Exploratory Analysis ...
        pass  # Replace this with your actual code for exploratory analysis

    elif app_mode == "View Routes":
        sled_routes = read_submission_file('submission.txt')
        sled_choice = st.sidebar.selectbox('Select a Sledge:', list(sled_routes.keys()))

        if sled_choice:
            # Load the dataset
            df = pd.read_csv('Datasets/2_Delft.txt', sep=' ', skiprows=1, header=None)
            df.columns = ['Start_Horizontal', 'Start_Vertical', 'Finish_Horizontal', 'Finish_Vertical', 'Earliest_Start', 'Latest_Finish']

            st.subheader(f'Route for {sled_choice}:')
            plot_sledge_route(sled_routes[sled_choice], df)

if __name__ == "__main__":
    main()

