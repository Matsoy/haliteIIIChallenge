##########################
### Last submitted bot ###
##########################

#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

import numpy as np
import scipy.stats as st

# This library contains constant values.
from hlt import constants, entity

from heapq import heappop, heappush

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

import math 

from hlt import Direction, Position

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

def in_enemy_area(position):
    if len(enemies_ships):
        return game_map.calculate_distance(
            min(
                enemies_ships,
                key=lambda s: game_map.calculate_distance(s.position, position)
            ).position
            ,
            position
        ) <= 1

    return False


def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    graph = {
        (i, j): [] for j in range(width) for i in range(height) if not maze[i][j]
    }

    for row, col in graph.keys():
        if (
            row < height - 1 
            and 
            not maze[row + 1][col]
        ):
            graph[(row, col)].append(
                (
                    "s",
                    (row + 1, col)
                )
            )
            graph[(row + 1, col)].append(
                (
                    "n", 
                    (row, col)
                )
            )

        if (
            col < width - 1 
            and 
            not maze[row][col + 1]
        ):
            graph[(row, col)].append(
                (
                    "e",
                    (row, col + 1)
                )
            )
            graph[(row, col + 1)].append(
                (
                    "w", 
                    (row, col)
                )
            )

    return graph


def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def find_path_astar(maze, start, goal):
    pr_queue = []
    heappush(
        pr_queue,
        (
            0 + heuristic(start, goal),
            0,
            "",
            start
        )
    )
    visited = set()
    graph = maze2graph(maze)

    while pr_queue:
        _, cost, path, current = heappop(pr_queue)

        if current == goal:
            return path

        if current in visited:
            continue

        visited.add(current)

        for direction, neighbour in graph[current]:
            heappush(
                pr_queue,
                (
                    cost + heuristic(neighbour, goal),
                    cost + 1,
                    path + direction, neighbour
                )
            )

    return "o"


def get_direction_from_waze_recursive(maze, maze_width, maze_height, origin, start, goal, start_position):
    # update parameters for the next loop
    new_maze_width = maze_width + 2
    new_maze_height = maze_height + 2
    new_origin = origin - Position(1, 1)
    new_start = (start[0] + 1, start[1] + 1)
    new_goal = (goal[0] + 1, goal[1] + 1)

    new_maze = np.append(
        maze,
        [[0]] * maze_height,
        axis=1
    )

    new_maze = np.insert(
        new_maze,
        0,
        0,
        axis=1
    )

    new_maze = np.append(
        new_maze,
        [[0] * new_maze_width],
        axis=0
    )

    new_maze = np.insert(
        new_maze,
        0,
        0,
        axis=0
    )

    impossible = True

    # the top line and the bottom line
    for i in range(0, new_maze_width):
        position_tmp1 = game_map.normalize(Position(new_origin.x + i, new_origin.y))
        if (
            game_map[position_tmp1].is_occupied
            or
            (
                game_map.calculate_distance(start_position, position_tmp1) == 1
                and
                in_enemy_area(position_tmp1)
            )
        ):
            impossible = False
            new_maze[0][i] = 1

        position_tmp2 = game_map.normalize(Position(new_origin.x + i, new_origin.y + new_maze_height - 1))
        if (
            game_map[position_tmp2].is_occupied
            or
            (
                game_map.calculate_distance(start_position, position_tmp2) == 1
                and
                in_enemy_area(position_tmp2)
            )
        ):
            impossible = False
            new_maze[new_maze_height - 1][i] = 1

    # the rest of the left column the right column
    for j in range(1, new_maze_height - 1):
        position_tmp1 = game_map.normalize(Position(new_origin.x, new_origin.y + j))
        if (
            game_map[position_tmp1].is_occupied
            or
            (
                game_map.calculate_distance(start_position, position_tmp1) == 1
                and
                in_enemy_area(position_tmp1)
            )
        ):
            impossible = False
            new_maze[j][0] = 1

        position_tmp2 = game_map.normalize(Position(new_origin.x + new_maze_width - 1, new_origin.y + j))
        if (
            game_map[position_tmp2].is_occupied
            or
            (
                game_map.calculate_distance(start_position, position_tmp2) == 1
                and
                in_enemy_area(position_tmp2)
            )
        ):
            impossible = False
            new_maze[j][new_maze_width - 1] = 1
    
    path = find_path_astar(
        new_maze, 
        new_start,
        new_goal
    )

    if (
        path == "o"
        and
        (
            new_maze_width < WIDTH / 3
            or
            new_maze_height < WIDTH / 3
        )
    ):
        if impossible:
            return "o"
        return get_direction_from_waze_recursive(new_maze, new_maze_width, new_maze_height, new_origin, new_start, new_goal, start_position)
    else:
        return path


def get_direction_from_waze(start_position, goal_position):
    if (
        game_map[goal_position].is_occupied
        or
        start_position == goal_position
    ):
        return "o"   

    resulting_position = abs(start_position - goal_position)
    maze_width = min(
        resulting_position.x,
        WIDTH - resulting_position.x
    ) + 1
    maze_height = min(
        resulting_position.y,
        WIDTH - resulting_position.y
    ) + 1

    direction_to_position = {
        "w" : {
            "start" : (0, maze_width - 1),
            "goal" : (maze_height - 1, 0)
        },
        "sw" : {
            "start" : (0, maze_width - 1),
            "goal" : (maze_height - 1, 0)
        },
        "s" : {
            "start" : (0, 0),
            "goal" : (maze_height - 1, maze_width - 1)
        },
        "e" : {
            "start" : (0, 0),
            "goal" : (maze_height - 1, maze_width - 1)
        },
        "es" : {
            "start" : (0, 0),
            "goal" : (maze_height - 1, maze_width - 1)
        },
        "n" : {
            "start" : (maze_height - 1, 0),
            "goal" : (0, maze_width - 1) 
        },
        "en" : {
            "start" : (maze_height - 1, 0),
            "goal" : (0, maze_width - 1) 
        },
        "nw" : {
            "start" : (maze_height - 1, maze_width - 1),
            "goal" : (0, 0)
        }
    }

    # initialize the matrix
    maze = [0] * maze_height
    for i in range(0, maze_height):
        maze[i] = [0] * maze_width

    start_to_goal_direction = ''.join(
        sorted(
            list(
                map(
                    lambda d: Direction.convert(d),
                    game_map.get_unsafe_moves(start_position, goal_position)
                )
            )
        )
    )

    start = direction_to_position[start_to_goal_direction]["start"]
    goal = direction_to_position[start_to_goal_direction]["goal"]
    origin = start_position - Position(start[1], start[0])

    # set 1 if the there is a ship
    for i in range(0, maze_width):
        for j in range(0, maze_height):
            position_tmp = game_map.normalize(Position(origin.x + i, origin.y + j))
            if (
                game_map[position_tmp].is_occupied
                and
                not (j, i) in [start, goal]
                or 
                (
                    game_map.calculate_distance(start_position, position_tmp) == 1
                    and
                    in_enemy_area(position_tmp)
                )
            ):
                maze[j][i] = 1

    path = find_path_astar(
        maze, 
        start,
        goal
    )

    if path == "o":
        return get_direction_from_waze_recursive(maze, maze_width, maze_height, origin, start, goal, start_position)
    else:
        return path        


def get_ships_around(from_position, count_enemies = False, count_allies = False , area = None):
    if area is None:
        area = constants.INSPIRATION_RADIUS
    
    count = dict()

    if count_enemies:
        count["enemies"] = len(
            list(
                filter(
                    lambda ship: (
                        game_map.calculate_distance(ship.position, from_position) 
                        <=
                        area
                    ), 
                    enemies_ships
                )
            )
        )


    if count_allies:
        count["allies"] = len(
            list(
                filter(
                    lambda ship: (
                        game_map.calculate_distance(ship.position, from_position) 
                        <=
                        area
                    ), 
                    me.get_ships()
                )
            )
        )

    return count


def get_extraction(from_position = None, with_inspiration = True):
    if from_position is None:
        return max(1, int(math.ceil(min_halite_to_stay * (1 / constants.EXTRACT_RATIO))))

    # extracted halite per default without inspiration
    extracted_halite = int(math.ceil(game_map[from_position].halite_amount * (1 / constants.EXTRACT_RATIO)))

    if (
        with_inspiration
        and
        constants.INSPIRATION_ENABLED
        and 
        get_ships_around(from_position, True)["enemies"]
        >= 
        constants.INSPIRATION_SHIP_COUNT
    ):
        extracted_halite *= int((constants.INSPIRED_BONUS_MULTIPLIER + 1))

    return extracted_halite


def numerical_superiority(from_position, area = 3):
    coeff = 1.5 if len(me.get_ships()) < len(enemies_ships) * 1.2 else 1
    ships_around = get_ships_around(from_position, True, True, area) 
    
    return (ships_around["allies"] - 1) > (ships_around["enemies"] - 1) * coeff


def get_best_dropoff(from_position):
    shipyard_and_dropoffs = [me.shipyard] + me.get_dropoffs()
    closest_dropoff = get_closest_shipyard_or_dropoff(from_position)

    # filters the dropoffs whose cost of travel is close to that of the nearest dropoff
    filtered_shipyard_and_dropoffs = list(
        filter(
            lambda i: (
                game_map.calculate_distance(from_position, closest_dropoff.position) * 1.5
                >=
                game_map.calculate_distance(i.position, from_position)
            ), 
            shipyard_and_dropoffs
        )
    )

    return max(
        filtered_shipyard_and_dropoffs, 
        key=lambda i: get_halite_around(i.position, 5)
    )


def get_halite_around(from_position, area):
    total_halite_around = 0

    for i in range(from_position.x - area, from_position.x + area + 1):
        for j in range(from_position.y - area, from_position.y + area + 1):
            total_halite_around += game_map[Position(i, j)].halite_amount
    
    return total_halite_around


def count_available_halite():
    total_halite = 0

    for x in range(0, WIDTH):
        for y in range(0, WIDTH):
            total_halite += game.game_map[Position(x, y)].halite_amount

    return total_halite


def update_halite_collected_ratio():
    return 1 - (count_available_halite() / HALITE_AT_THE_BEGINNING)
    

def get_closest_shipyard_or_dropoff(from_position, take_into_account_other_players = False, without_position = None):
    shipyard_and_dropoffs = [me.shipyard] + me.get_dropoffs()
    
    if take_into_account_other_players:
        for player_id, player_object in game.players.items():
            if not player_object == me:
                shipyard_and_dropoffs += [player_object.shipyard]

    return min(
        # remove the entity at without_position
        filter(
            lambda i: True if without_position is None else not i.position == without_position, 
            shipyard_and_dropoffs
        ),
        key=lambda j: game_map.calculate_distance(j.position, from_position)
    )


def can_spawn_dropoff(area = 5):
    if not (
        halite_collected_ratio < 0.65
        and
        len(me.get_dropoffs()) < MAX_DROPOFFS[WIDTH][NB_PLAYERS]
        and
        (game.turn_number / constants.MAX_TURNS) <= 0.7
        and
        len(me.get_ships()) >= 15
    ):
        return False

    shipyard_and_dropoffs = [me.shipyard] + me.get_dropoffs()
    for s in shipyard_and_dropoffs:
        halite_around = get_halite_around(s.position, area)
        average_halite_around = halite_around / ((area + 1) * (area + 1))
        if (
            average_halite_around / 3.5
            > 
            count_available_halite() / (WIDTH * WIDTH)
        ):
            return False
        
    global stop_spending_halite
    stop_spending_halite = False
    anticipated_dropoffs.clear()    

    return True    
    

def apply_movement(ship, command):
    command_queue.append(command)

    # indicates that the ship has played
    ship.has_already_played = True

    # save the next positions of allied ships
    # if this it a "move" command
    if command[0] == "m":
        direction = COMMAND_TO_DIRECTION[
            str(command[-1:])
        ]
        next_position = game_map.normalize(ship.position.directional_offset(direction))
        next_positions.append(game_map.normalize(next_position))
    # if this is a "construct" command
    elif command[0] == "c":
        next_positions.append(game_map.normalize(ship.position))


    if not swapping:
        # if this it a "move" command
        if command[0] == "m":
            # mark the former position as safe for other allied ships
            direction = COMMAND_TO_DIRECTION[
                str(command[-1:])
            ]
            next_position = game_map.normalize(ship.position.directional_offset(direction))
            # if the ship move on another position
            if not next_position == game_map.normalize(ship.position):
                game_map[ship.position].mark_unsafe(None)
        # if this is a "construct" command
        elif command[0] == "c":
            game_map[ship.position].mark_unsafe(None)
    

def gaussian_kernel(gaussian_len=3, sigma=3):
    """
    Returns a 2D Gaussian kernel array.

    :param gaussian_len: The kernel length. Only an odd number
    :param sigma: Sigma, the strength of the blur.
    :return: A 2D Gaussian kernel array.
    """

    interval = (2*sigma+1.)/(gaussian_len)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., gaussian_len+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def blur(gaussian_len=10, sigma=3):
    # get the gaussian_kernel
    kernel = gaussian_kernel(gaussian_len, sigma)

    offset = int((gaussian_len - 1) / 2)
    total_width = WIDTH + 2 * offset    

    blurred_matrix = [0] * total_width

    # fill the outside
    for x in range(0, total_width):
        blurred_matrix[x] = [0] * total_width
        for y in range(0, total_width):
            # if it's the left
            if x < offset:
                blurred_matrix[x][y] = float(game_map[Position(total_width - x - 1, y)].halite_amount)
            
            # if it's the right
            elif x > offset + WIDTH:
                blurred_matrix[x][y] = float(game_map[Position(x - offset - WIDTH, y)].halite_amount)

            # if it's the up
            elif y < offset:
                blurred_matrix[x][y] = float(game_map[Position(x, total_width - y - 1)].halite_amount)
            
            # if it's the down
            elif y > offset + WIDTH:
                blurred_matrix[x][y] = float(game_map[Position(x, y - offset - WIDTH)].halite_amount)
            
            # else, it's the center
            else:
                blurred_matrix[x][y] = float(game_map[Position(x, y)].halite_amount)

    arraylist = []
    for y in range(gaussian_len):
        temparray = np.copy(blurred_matrix)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(gaussian_len):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)

    # remove offsets
    final_blurred_matrix = arraylist_sum[
        offset:offset + WIDTH,
        offset:offset + WIDTH
    ]
        
    return final_blurred_matrix


def custom_naive_navigate(ship, destination, crash_into_shipyard, simulate, position_to_match):
    """
    Returns a singular safe move towards the destination.

    :param ship: The ship to move.
    :param destination: Ending position
    :param crash_into_shipyard: True if the ship must crash into the shipyard, False otherwise
    :param simulate: True if should mark the cell as unsafe for the next ships, False otherwise
    :param position_to_match: if we have to see if this position is attainable in 1 movement (to swap with another boat)
    :return: A direction.
    """

    available_positions = []
    blocked_positions_because_of_an_enemy = []

    for direction in game_map.get_unsafe_moves(ship.position, destination):
        target_pos = game_map.normalize(ship.position.directional_offset(direction))

        close_enemies_ships = [
            a for a in enemies_ships 
            if game_map.calculate_distance(a.position, target_pos) <= 1
        ]

        close_allied_ships = [
            a for a in me.get_ships() 
            if game_map.calculate_distance(a.position, target_pos) <= 1
        ]
        close_allied_ships.remove(ship)

        # if the ship must crash into the shipyard and if the ship is 1 displacement of the shipyard
        if (
            crash_into_shipyard 
            and 
            game_map.calculate_distance(ship.position, destination) == 1
        ):
            available_positions.append([target_pos, direction])
        else:
            # save positions blocked by an enemy
            # if this position is in the area of ​​an enemy
            # and that an allied ship has not already moved to that position
            if (
                len(close_enemies_ships) 
                and
                not target_pos in next_positions
            ):
                blocked_positions_because_of_an_enemy.append(target_pos)

            # else, if the position isn't occupied and not close to the enemy
            elif not game_map[target_pos].is_occupied:
                available_positions.append([target_pos, direction])
            # else, we check if the position is occupied by an allied ship
            else:
                for allied_ship in close_allied_ships:
                    # if the ship has not already played and if it's at this position
                    if (
                        not allied_ship.has_already_played
                        and
                        allied_ship.position == target_pos
                    ):
                        
                        if (
                            not position_to_match is None
                            and
                            position_to_match == allied_ship.position
                            and
                            simulate
                            ):
                            return direction

                        elif not simulate:
                            # get the next movement of the allied ship in a simulate turn
                            allied_command = get_next_movement(allied_ship, True, game_map.normalize(ship.position))
                            allied_direction = COMMAND_TO_DIRECTION[
                                str(allied_command[-1:])
                            ]
                            next_simulated_allied_position = game_map.normalize(target_pos.directional_offset(allied_direction))

                            # if these ships can swap their position 
                            if next_simulated_allied_position == ship.position:
                                global swapping
                                swapping = True
                                # mark these position as unsafe
                                game_map[ship.position].mark_unsafe(allied_ship)
                                game_map[target_pos].mark_unsafe(ship)
                                # apply the movement of the allied ship
                                apply_movement(
                                    allied_ship,
                                    allied_ship.move(allied_direction)
                                )

                                # and returns that of the current ship
                                return direction
                            
                            # if the allied ship will move by releasing the position
                            elif next_simulated_allied_position != allied_ship.position:
                                game_map[next_simulated_allied_position].mark_unsafe(allied_ship)
                                # apply the movement of the allied ship
                                apply_movement(
                                    allied_ship,
                                    allied_ship.move(allied_direction)
                                )
                                # important to mark unsafe AFTER applying the movement of the allied ship
                                game_map[target_pos].mark_unsafe(ship)
                                return direction                             

    if len(available_positions):
        sorted_positions_by_ascending_halite = sorted(
            available_positions, 
            key=lambda p: game_map[p[0]].halite_amount, 
            reverse=False
        ) 

        cheapest_positions = [sorted_positions_by_ascending_halite[0]]
        
        # if the 2nd position with the least halite costs the same cost
        if (
            len(available_positions) == 2 
            and 
            int(game_map[sorted_positions_by_ascending_halite[0][0]].halite_amount * (1 / constants.MOVE_COST_RATIO))
            ==
            int(game_map[sorted_positions_by_ascending_halite[1][0]].halite_amount * (1 / constants.MOVE_COST_RATIO))
        ):
            cheapest_positions.append(sorted_positions_by_ascending_halite[1])
        
        chosen_position = random.choice(cheapest_positions)
        
        if not simulate:
            game_map[chosen_position[0]].mark_unsafe(ship)
        return chosen_position[1]

    for blocked_position in blocked_positions_because_of_an_enemy:
        if (
            numerical_superiority(blocked_position)
            and
            not (
                game_map[blocked_position].ship in me.get_ships()
                and 
                not game_map[blocked_position].ship.has_already_played
            )
        ):
            direction = game_map.get_unsafe_moves(ship.position, blocked_position)[0]
            if not simulate:
                game_map[blocked_position].mark_unsafe(ship)
            return direction

    if (
        not position_to_match is None
        and
        # the ship refuse if it's in ships_constructing_dropoff
        not game_map[position_to_match].ship.id in ships_constructing_dropoff
        and
        # the ship refuse if it's in ships_coming_back and if the other ship is not is ships_constructing_dropoff
        (
            not ship in ships_coming_back
            or
            game_map[position_to_match].ship.id in ships_constructing_dropoff
        )
    ):
        direction = game_map.get_unsafe_moves(ship.position, position_to_match)[0]
        return direction

    waze_direction = get_direction_from_waze(
        game_map.normalize(ship.position),
        game_map.normalize(destination)
    )[0]

    direction = COMMAND_TO_DIRECTION[
        waze_direction
    ]
    
    if not simulate:
        game_map[game_map.normalize(ship.position.directional_offset(direction))].mark_unsafe(ship)
    
    return direction


def scan_area(area, position, find_one = None):       
    """
    Recursive function. Returns the position with the maximum of halite around a given position
    :param area: the area of the zone to scope 
    :return: the Position object 
    """


    # the most recent shipyard
    try:
        most_recent_shipyard = dropoffs_history[
            min(
                dropoffs_history.keys(),
                key=lambda k: dropoffs_history[k]["turns_ago"]
            )
        ]

        ship_id = game_map[position].ship.id

        if (
            most_recent_shipyard["turns_ago"] < 30
            and
            len(most_recent_shipyard["ships_in_area"]) < 5
            and
            not ship_id in most_recent_shipyard["ships_in_area"]
        ):
            distance_from_the_nearest_dropoff = game_map.calculate_distance(
                get_closest_shipyard_or_dropoff(
                    position,
                    False,
                    most_recent_shipyard["position"]
                ).position,
                most_recent_shipyard["position"]
            )

            distance_from_the_ship = game_map.calculate_distance(
                position,
                most_recent_shipyard["position"]
            )

            if distance_from_the_nearest_dropoff >= distance_from_the_ship:
                # if the ship is now in the area
                if distance_from_the_ship <= 5:
                    most_recent_shipyard["ships_in_area"].append(ship_id)
                return most_recent_shipyard["position"]

    except ValueError:
        pass

    if (
        not find_one is None
        and
        area - find_one["area"] > 5
    ):
        return find_one["position"]

    all_options = [];
    
    ########################################################################## 
    # example for area = 3
    # with <X> : position at the right distance from <o>
    #       x   0   1   2   3   4   5   6
    #   y
    #   0       .   .   .   X   .   .   .
    #   1       .   .   X   .   X   .   .
    #   2       .   X   .   .   .   X   .
    #   3       X   .   .   o   .   .   X
    #   4       .   X   .   .   .   X   .
    #   5       .   .   X   .   X   .   .
    #   6       .   .   .   X   .   .   .

    # if like the example, add Position(0, 3) and Position(6, 3) because <offset_y> would be 0
    all_options.append(game_map.normalize(Position(position.x - area, position.y)))
    all_options.append(game_map.normalize(Position(position.x + area, position.y)))

    offset_y = 1

    # for each x, add the 2 positions that are at the right distance
    for i in range(position.x - area + 1, position.x + area):
        all_options.append(game_map.normalize(Position(i, position.y - offset_y)))
        all_options.append(game_map.normalize(Position(i,  position.y + offset_y)))

        if offset_y < area:
            offset_y += 1
        else:
            offset_y -= 1
    
    # remove if is_occupied and no enough halite 
    # and sort by halite
    sorted_filtered_options = sorted(
        list(
            filter(
                lambda opt: (
                    not game_map[opt].is_occupied
                    and
                    int(game_map[opt].halite_amount * (1 / constants.EXTRACT_RATIO)) 
                    >= 
                    int(get_extraction()) 
                ),
                all_options
            )
        ),
        key=lambda opt2: game_map[opt2].halite_amount, 
        # key=lambda opt2: get_halite_around(opt2, 3),
        reverse=True
    ) 


    if len(sorted_filtered_options):
        if find_one is None:
            return scan_area(
                area + 1,
                position,
                {
                    "position" : sorted_filtered_options[0],
                    "area" : area,
                    "extraction" : int(game_map[sorted_filtered_options[0]].halite_amount * (1 / constants.EXTRACT_RATIO)) 
                }
            )
        else:
            # if the new position is better
            if int(game_map[sorted_filtered_options[0]].halite_amount * (1 / constants.EXTRACT_RATIO)) > find_one["extraction"] * 50:
                return scan_area(
                    area + 1,
                    position,
                    {
                        "position" : sorted_filtered_options[0],
                        "area" : find_one["area"],
                        "extraction" : int(game_map[sorted_filtered_options[0]].halite_amount * (1 / constants.EXTRACT_RATIO)) 
                    }
                )
            else:
                return scan_area(
                    area + 1,
                    position,
                    # [position, distance, extraction]
                    find_one
                )

    else:
        # recall scan_area with area + 1
        if area < WIDTH:
            return scan_area(area + 1, position, find_one)
        else:
            return position
    
        
def have_enough_halite_to_move(ship):
    return (
        ship.halite_amount 
        >=
        int(game_map[ship.position].halite_amount * (1 / constants.MOVE_COST_RATIO))
    )

    
def get_next_movement(current_ship, simulate, position_to_match = None):
    """
    Returns the next movement of the current ship
    :return: a move to move this ship
    """

    # if the ship have no enough halite to move
    if not have_enough_halite_to_move(current_ship):
        return current_ship.stay_still()
    
    # if the ship must move towards an anticipated dropoff
    if (
        stop_spending_halite 
        and
        current_ship.id in anticipated_dropoffs
    ):
        if (
            game_map.calculate_distance(
                current_ship.position, 
                anticipated_dropoffs[current_ship.id]
            ) == 1
        ):
            return current_ship.stay_still() 

        max_pos = anticipated_dropoffs[current_ship.id]
        direction = custom_naive_navigate(current_ship, max_pos, False, simulate, position_to_match)
        movement = current_ship.move(direction)

        return current_ship.stay_still()



    # if the ship is at 1 displacement of the shipyard
    elif (
        farthest_ship_coming_back_because_of_the_time 
        and 
        game_map.calculate_distance(current_ship.position, get_closest_shipyard_or_dropoff(current_ship.position).position) == 1
        ):
        direction = custom_naive_navigate(current_ship, get_closest_shipyard_or_dropoff(current_ship.position).position, True, simulate, position_to_match)
        movement = current_ship.move(direction)



    # if it's time to coming back 
    elif (
        game_map.calculate_distance(current_ship.position, get_closest_shipyard_or_dropoff(current_ship.position).position) 
        >
        constants.MAX_TURNS - game.turn_number - OFFSET_BEFORE_COMING_BACK
    ):
        # if it's time to coming back but the current cell is interesting
        if(
            current_ship.halite_amount <= constants.MAX_HALITE * MIN_PERCENT_BEFORE_COMING_BACK
            and
            game_map.calculate_distance(current_ship.position, get_closest_shipyard_or_dropoff(current_ship.position).position) 
            <= 
            constants.MAX_TURNS - game.turn_number - int(OFFSET_BEFORE_COMING_BACK / 2)
            and 
            int(game_map[current_ship].halite_amount * (1 / constants.EXTRACT_RATIO)) * 10
            >=
            get_extraction()
        ):
            return current_ship.stay_still()
        
        direction = custom_naive_navigate(current_ship, get_closest_shipyard_or_dropoff(current_ship.position).position, True, simulate, position_to_match)
        movement = current_ship.move(direction)

    # in ships_coming_back
    elif current_ship in ships_coming_back:
        if (
            current_ship.halite_amount < constants.MAX_HALITE
            and 
            int(game_map[current_ship].halite_amount * (1 / constants.EXTRACT_RATIO))
            >=
            get_extraction()
        ):
            return current_ship.stay_still()
    
        # else if the ship is full
        else:
            # if the ship is one cell away from the shipyard and the shipyard is occupied
            if (
                game_map.calculate_distance(current_ship.position, get_closest_shipyard_or_dropoff(current_ship.position).position) == 1
                and
                game_map[get_closest_shipyard_or_dropoff(current_ship.position).position].is_occupied
                and
                len(enemies_ships)
            ):
                go_into_the_enemy = False
                for ship in enemies_ships:
                    # if there is a ship that is to the enemy
                    if ship.position == get_closest_shipyard_or_dropoff(current_ship.position).position:
                        go_into_the_enemy = True
                        break

                direction = custom_naive_navigate(current_ship, get_closest_shipyard_or_dropoff(current_ship.position).position, go_into_the_enemy, simulate, position_to_match)
                movement = current_ship.move(direction)
            else:
                direction = custom_naive_navigate(current_ship, get_best_dropoff(current_ship.position).position, False, simulate, position_to_match)
                movement = current_ship.move(direction)

    # normal ship
    else:
        simulated_halite_amount = game_map[current_ship].halite_amount

        if constants.INSPIRATION_ENABLED:
            enemy_ships_in_radius_count = 0
            sorted_enemies_ships_by_ascending_distance = sorted(
                enemies_ships,
                key=lambda s: game_map.calculate_distance(s.position, current_ship.position),
                reverse=False
            )  

            for enemy_ship in sorted_enemies_ships_by_ascending_distance:
                if (
                    game_map.calculate_distance(enemy_ship.position, current_ship.position) 
                    <= 
                    constants.INSPIRATION_RADIUS
                ):
                    enemy_ships_in_radius_count += 1
                else:
                    break
            
            if enemy_ships_in_radius_count >= constants.INSPIRATION_SHIP_COUNT:
                simulated_halite_amount = int(simulated_halite_amount * (constants.INSPIRED_BONUS_MULTIPLIER + 1))

        # if more than min_halite_to_stay halite in this case
        if (
            int(simulated_halite_amount * (1 / constants.EXTRACT_RATIO))
            >= 
            get_extraction()
        ):
            return current_ship.stay_still()

        max_pos = scan_area(1, current_ship.position)
        direction = custom_naive_navigate(current_ship, max_pos, False, simulate, position_to_match)
        movement = current_ship.move(direction)

    return movement


def can_spawn_ship():
    """
    Returns true if we have to spawn a ship, false otherwise
    :return: true if we have to spawn a ship, false otherwise
    """

    return (
        (        
            (game.turn_number / constants.MAX_TURNS) <= SPAWN_SHIP_TURN
            and 
            not game_map[me.shipyard].is_occupied
            and 
            not game_map[me.shipyard].position in next_positions
        )
        and
        (
            (
                not stop_spending_halite
                and
                me.halite_amount >= constants.SHIP_COST
            )
            or
            (
                stop_spending_halite
                and
                me.halite_amount >= constants.SHIP_COST + constants.DROPOFF_COST

            )
        )
        and
        len(me.get_ships()) <= len(enemies_ships) * 1.75
        and
        halite_collected_ratio < 0.55
    )


def will_have_enough_halite_to_create_a_dropoff():
    # number of ships that will have returned
    nb_ships = len(ships_coming_back)

    # if the available halite will be sufficient
    if (
        nb_ships * constants.MAX_HALITE + me.halite_amount
        >=
        constants.DROPOFF_COST
    ):
        return True
    else:
        return False


def move_ships():
    reset_stop_spending_halite = False

    # for each ship, move it
    for ship in (ships_constructing_dropoff + ships_coming_back + ships_on_shipyard + ships_default):
        global swapping
        swapping = False
        # if the ship has not already played
        if not ship.has_already_played:
            # if the boat is on an anticipated dropoff
            # and if we have enough halite
            if (
                ship.id in anticipated_dropoffs
                and
                game_map.calculate_distance(
                    ship.position, 
                    anticipated_dropoffs[ship.id]
                ) <= 2
                and
                me.halite_amount >= constants.DROPOFF_COST
                and 
                # whether this cell has no structure
                not game_map[ship.position].has_structure
            ):
                # if we also have enough halite to spawn a ship
                if me.halite_amount >= (constants.DROPOFF_COST + constants.SHIP_COST):
                    stop_spending_halite = False

                reset_stop_spending_halite = True
                
                apply_movement(
                    ship,
                    ship.make_dropoff()
                )
            else:
                apply_movement(
                    ship,
                    get_next_movement(ship, False)
                )

    return reset_stop_spending_halite


def play_turn():
    """
    Play a turn while moving ships
    """
    global stop_spending_halite 

    clear_anticipated_dropoffs = False
    # delete the ship if it no longer exists
    for ship_id, anticipated_dropoff_position in anticipated_dropoffs.items():
        if not ship_id in me._ships:
            clear_anticipated_dropoffs = True
    
    if clear_anticipated_dropoffs:
        anticipated_dropoffs.clear()

    # if will have enough halite to create a dropoff
    # looking for the best positions to construct a dropoff
    if (
        not len(anticipated_dropoffs)
        and
        can_spawn_dropoff() 
        and 
        will_have_enough_halite_to_create_a_dropoff()
    ):
        # get all positions
        all_positions = []

        enemy_shipyards = []
        for player_id, player_object in game.players.items():
            if not player_object == me:
                enemy_shipyards.append(player_object.shipyard)
        
        farthest_shipyard = max(
            enemy_shipyards,
            key=lambda s: game_map.calculate_distance(me.shipyard.position, s.position)
        ) 

        # use a step for large maps, in order to avoid a timeout  
        step = 2 if WIDTH >= 56 else 1

        for i in range (0, WIDTH, step):
            for j in range (0, WIDTH, step):
                position_tmp = Position(i, j)
                distance_from_closest_shipyard_or_dropoff = game_map.calculate_distance(
                    position_tmp,
                    get_closest_shipyard_or_dropoff(position_tmp).position
                ) 
                if (
                    distance_from_closest_shipyard_or_dropoff 
                    >= 
                    WIDTH / 4
                    and
                    distance_from_closest_shipyard_or_dropoff
                    <=
                    WIDTH / 2
                    and
                    not (
                        min(me.shipyard.position.x, farthest_shipyard.position.x) 
                        < 
                        position_tmp.x
                        and 
                        max(me.shipyard.position.x, farthest_shipyard.position.x) 
                        > 
                        position_tmp.x
                        and 
                        min(me.shipyard.position.y, farthest_shipyard.position.y) 
                        < 
                        position_tmp.y
                        and 
                        max(me.shipyard.position.y, farthest_shipyard.position.y) 
                        > 
                        position_tmp.y
                    )
                ):
                    all_positions.append(position_tmp)
        
        # sort all positions by halite around
        sorted_all_positions_by_halite_around = sorted(
            all_positions,
            key=lambda opt: get_halite_around(opt, 5),
            reverse=True
        ) 

        # filters positions that are 30% worse than the best position 
        filtered_sorted_all_positions_by_halite_around = [
            i for i in sorted_all_positions_by_halite_around if (
                get_halite_around(i, 5) * 1.3 
                >=
                get_halite_around(sorted_all_positions_by_halite_around[0], 5)
            )
        ]

        # sort by distance
        sorted_all_positions_by_distance = sorted(
            filtered_sorted_all_positions_by_halite_around,
            key=lambda opt: game_map.calculate_distance(
                get_closest_shipyard_or_dropoff(opt).position,
                opt
            ),
            reverse=False
        ) 
        
        chosen_ship = None
        chosen_position = None

        # we only look at 5% of the best positions
        for eligible_position in filtered_sorted_all_positions_by_halite_around[:(int(len(all_positions)/20))]:
            for eligible_ship in ships_default:
                distance_tmp = game_map.calculate_distance(eligible_ship.position, eligible_position)
                if distance_tmp <= WIDTH:
                    # if it's the first ship in the perimeter
                    # or if it's closer than chosen_ship
                    if (
                        chosen_ship is None
                        or
                        (
                            distance_tmp
                            <
                            game_map.calculate_distance(chosen_ship.position, eligible_position)
                        )
                    ):
                        chosen_ship = eligible_ship
            
            # if we found a ship that could go to the position in the given number of turns
            if not chosen_ship is None:
                chosen_position = eligible_position
        
        if not chosen_position is None:
            stop_spending_halite = True
            anticipated_dropoffs[chosen_ship.id] = chosen_position

    reset_stop_spending_halite = move_ships()

    if can_spawn_ship():
        # spawn a ship (-1000 halite)
        command_queue.append(me.shipyard.spawn())
    
    if reset_stop_spending_halite:
        stop_spending_halite = False
        anticipated_dropoffs.clear()

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
    

def fill_ship_arrays():
    global ships_coming_back 

    unsorted_ships_coming_back = []
    unsorted_ships_default = []
    
    for ship in me.get_ships():
        ship.has_already_played = False
    
        # if this ship has to construct a dropoff
        if (ship.id in anticipated_dropoffs):
            ships_constructing_dropoff.append(ship)
        
        # time to coming back
        elif (
            ship.halite_amount > constants.MAX_HALITE * MIN_PERCENT_BEFORE_COMING_BACK
            or
            game_map.calculate_distance(ship.position, get_closest_shipyard_or_dropoff(ship.position).position) 
            > 
            constants.MAX_TURNS - game.turn_number - OFFSET_BEFORE_COMING_BACK
        ):
            unsorted_ships_coming_back.append(ship)

        else:
            is_in_ships_coming_back = False
            for ship2 in ships_coming_back:
                if ship.id == ship2.id:
                    is_in_ships_coming_back = True
                    break 
                
            # if it was a ship coming back during the last turn
            if is_in_ships_coming_back:
                if (
                    ship.halite_amount / constants.MAX_HALITE 
                    < 
                    MINIMUM_HALITE_TO_BECOME_A_DEFAULT_SHIP_AGAIN
                ): 
                    # it become a default ship
                    unsorted_ships_default.append(ship)
                else:
                    # it's still a ship coming back
                    unsorted_ships_coming_back.append(ship)
            elif ship.position == get_closest_shipyard_or_dropoff(ship.position).position:
                ships_on_shipyard.append(ship)
            else:
                unsorted_ships_default.append(ship)
        
    # firstly the ships closest to the shipyard
    ships_coming_back = sorted(
        unsorted_ships_coming_back,
        key=lambda s: game_map.calculate_distance(s.position, get_closest_shipyard_or_dropoff(ship.position).position),
        reverse=False
    )  
   
    # firstly the ships farthest to the shipyard
    global ships_default 
    ships_default = sorted(
        unsorted_ships_default, 
        key=lambda s: game_map.calculate_distance(s.position, get_closest_shipyard_or_dropoff(ship.position).position), 
        reverse=True
    )  

    global farthest_ship_coming_back_because_of_the_time
    farthest_ship_coming_back_because_of_the_time = (
        len(ships_coming_back)
        and
        game_map.calculate_distance(ships_coming_back[len(ships_coming_back) - 1].position, get_closest_shipyard_or_dropoff(ship.position).position) 
        >
        constants.MAX_TURNS - game.turn_number - OFFSET_BEFORE_COMING_BACK)
    
    for ship in reversed(ships_default):
        # the list being sorted by increasing distance, we break the loop if the distance is greater than 1
        if game_map.calculate_distance(ship.position, get_closest_shipyard_or_dropoff(ship.position).position) > 1:
            break

        # if the ship is 1 displacement of the shipyard
        # and if the farthest ship starts coming back 
        # if it's soon time to coming back 
        if farthest_ship_coming_back_because_of_the_time:
            # add the ship to the first place of the ships coming back
            ships_coming_back.insert(0, ship)
            # deletes the ship from the default list
            ships_default.remove(ship)


def update_min_halite_to_stay():
    # total of halite on the map
    total_halite = count_available_halite()

    # average halite per cell 
    avg_halite_per_cell = int(total_halite / (WIDTH * WIDTH))

    global min_halite_to_stay
    min_halite_to_stay = (avg_halite_per_cell / 2) if (avg_halite_per_cell < DEFAULT_MIN_HALITE_TO_STAY * 0.8 ) else DEFAULT_MIN_HALITE_TO_STAY


def init_default_min_halite_to_stay():
    # total of halite on the map
    total_halite = count_available_halite()

    # average halite per cell 
    avg_halite_per_cell = total_halite / (WIDTH * WIDTH)

    return int(avg_halite_per_cell / 2.5)


def update_enemies_ships():
    for player_id, player_object in game.players.items():
        if not player_object == me:
            for ship in player_object.get_ships():
                enemies_ships.append(ship)


def update_dropoffs_history(area):
    for dropoff in me.get_dropoffs():
        if not dropoff.id in dropoffs_history:
            dropoffs_history[dropoff.id] = {
                "position" : dropoff.position,
                # number of turns since creation
                "turns_ago" : 0,
                # list of ids of ships that have passed in the area of the dropoff
                "ships_in_area" : list(
                    map(
                        lambda s: s.id,
                        list(
                            filter(
                                lambda ship: (
                                    game_map.calculate_distance(ship.position, dropoff.position) 
                                    <=
                                    area
                                ), 
                                me.get_ships()
                            )
                        )
                    )
                )
            }

        else:
            dropoffs_history[dropoff.id]["turns_ago"] += 1

        halite_around = get_halite_around(dropoff.position, area)
        nb_cells = (2 * area + 1) * (2 * area + 1)
        dropoffs_history[dropoff.id]["average_halite_around"] = halite_around / nb_cells

        # { "allies", "enemies" }
        dropoffs_history[dropoff.id]["ships_around"] = get_ships_around(
            dropoff.position,
            True,
            True,
            area
        )


global me
global game_map
global command_queue
# list of ships in the shipyard or in dropoffs
global ships_on_shipyard
# list of returning ships deposit the halite
global ships_coming_back 
ships_coming_back = []
# list of ships that have to go construct a dropoff
global ships_constructing_dropoff
# matrix of a Gaussian blur on the halite amount 
global gaussian_blur_map
# list of enemy ships
global enemies_ships
# True if the farthest ship starts coming back, False otherwise 
global farthest_ship_coming_back_because_of_the_time
# list of dropoffs that will be built in the current turn
global next_dropoffs
# ratio of collected halite
global halite_collected_ratio 
# True if we have to stop spending halite, False otherwise
global stop_spending_halite
stop_spending_halite = False
# ship id -> dropoff position
global anticipated_dropoffs
anticipated_dropoffs = dict()
global next_positions
global target_positions
target_positions = dict()
# id -> creation turn 
global dropoffs_history
dropoffs_history = dict()

# map width
WIDTH = game.game_map.width
# the default minimum halite that a position must have for a ship to stay on
DEFAULT_MIN_HALITE_TO_STAY = init_default_min_halite_to_stay()
# number of players
NB_PLAYERS = len(game.players)
# the minimum fill rate a ship must have before heading to the nearest shipyard or dropoff
MIN_PERCENT_BEFORE_COMING_BACK = 0.97
# the turn ratio before which new ships can be generated
SPAWN_SHIP_TURN = 0.6
# margin of error in turns for the return of ships at the end of the game
OFFSET_BEFORE_COMING_BACK = 18 if WIDTH >= 56 else 12
# cardinality -> direction
COMMAND_TO_DIRECTION = {
    "n" : (0, -1),
    "s" : (0, 1),
    "e" : (1, 0),
    "w" : (-1, 0),
    "o" : (0, 0)
}
ADJACENT_CARDINALITIES = {
    Direction.North : [Direction.West, Direction.East],
    Direction.South : [Direction.West, Direction.East],
    Direction.East : [Direction.North, Direction.South],
    Direction.West : [Direction.North, Direction.South]
}
# map width -> number of players -> maximum number of dropoffs 
MAX_DROPOFFS = dict([
    (32, dict([(2, 0), (4, 0)])),
    (40, dict([(2, 2), (4, 1)])),
    (48, dict([(2, 3), (4, 2)])),
    (56, dict([(2, 4), (4, 3)])),
    (64, dict([(2, 5), (4, 4)]))
])
# amount of halite available at launch of the game
HALITE_AT_THE_BEGINNING = count_available_halite()
MINIMUM_HALITE_TO_BECOME_A_DEFAULT_SHIP_AGAIN = 0.5

# print constants
logging.debug("SHIP_COST\t"                 + str(constants.SHIP_COST                ))
logging.debug("DROPOFF_COST\t"              + str(constants.DROPOFF_COST             ))
logging.debug("MAX_HALITE\t"                + str(constants.MAX_HALITE               ))
logging.debug("MAX_TURNS\t"                 + str(constants.MAX_TURNS                ))
logging.debug("EXTRACT_RATIO\t"             + str(constants.EXTRACT_RATIO            ))
logging.debug("MOVE_COST_RATIO\t"           + str(constants.MOVE_COST_RATIO          ))
logging.debug("INSPIRATION_ENABLED\t"       + str(constants.INSPIRATION_ENABLED      ))
logging.debug("INSPIRATION_RADIUS\t"        + str(constants.INSPIRATION_RADIUS       ))
logging.debug("INSPIRATION_SHIP_COUNT\t"    + str(constants.INSPIRATION_SHIP_COUNT   ))
logging.debug("INSPIRED_EXTRACT_RATIO\t"    + str(constants.INSPIRED_EXTRACT_RATIO   ))
logging.debug("INSPIRED_BONUS_MULTIPLIER\t" + str(constants.INSPIRED_BONUS_MULTIPLIER))
logging.debug("INSPIRED_MOVE_COST_RATIO\t"  + str(constants.INSPIRED_MOVE_COST_RATIO ))


while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()

    # instanciate global variables
    me = game.me
    game_map = game.game_map
    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    update_min_halite_to_stay()
    halite_collected_ratio = update_halite_collected_ratio()

    enemies_ships = []
    next_dropoffs = []
    next_positions = []
    target_positions.clear()

    update_enemies_ships()

    ships_on_shipyard = []
    ships_default = []
    ships_constructing_dropoff = []

    fill_ship_arrays()

    update_dropoffs_history(5)

    # play turn
    play_turn()