from mcts_node import MCTSNode
from random import choice, random
from math import sqrt, log, inf
from multiprocessing import Pool
from contextlib import closing
import time, copy

DEBUG = False
NUM_THREADS = 16


TIME_CONSTRAINT = True
#in seconds
max_time = 0.1

num_nodes = 10
explore_faction = 2

ROLLOUTS = 5
MAX_DEPTH = 5

THIS_IDENTITIY = 0
OTHER_IDENTITIY = 0


#board.legal_actions(state) returns the moves available in state.
#board.next_state(state, action) returns a new state constructed by applying action in state.
#board.is_ended(state) returns True if the game has ended in state and False otherwise.
#board.current_player(state)  returns the index of the current player in state.
#board.points_values(state)  returns a dictionary of the score for each player (eg {1:-1,2:1} for a second-player win).  Will return {1: 0, 2: 0} if the game is not ended.
#board.owned_boxes(state)  returns a dict with (Row,Column) keys; values indicate for each box whether player 1, 2, or 0 (neither) owns that box
#board.display(state)  returns a string representation of the board state.
#board.display_action(action)  returns a string representation of the game action.

#Never used
def find_root_move(best_move, node: MCTSNode):
    parent_move = best_move
    while node.parent:
        parent_move = node.parent_action
        node = node.parent
    return parent_move

#gets you the state of the current node using node parents and the initial state
def ravel_states(board, root_state, current_node: MCTSNode):
    moves = []
    parent_count = 0

    while current_node.parent:
        moves.append(current_node.parent_action)
        current_node = current_node.parent
        parent_count += 1
        if DEBUG and len(current_node.child_nodes) > 1:
            try:
                print("At {} level there are {} nodes".format(parent_count, len(current_node.child_nodes)))
                pass
            except:
                pass
    #if DEBUG: print("We were {} deep!".format(parent_count))
    current_state = root_state
    while moves:
        action = moves.pop()
        current_state = board.next_state(current_state, action)
    return current_state

#selection; navigates the tree node
def traverse_nodes(node: MCTSNode, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.

    Returns:        A node from which the next stage of the search can proceed.

    """
    #Implement UCT
    #Returns a random leaf node.

    # currently always goes for a leaf node.

    #the uct is using a different formula than given in class.
    #instead of using "current" and its "child" we are using "current" and its
    # "parent"
    # Technically there is no difference but it allows us to always have the current
    # visited count be non 0 (other wise comparing it to non visited children
    # resulted in division by 0
    def UCT(parent, current, identitity):
        global THIS_IDENTITIY

        #Similarly if the UCT is running from root, the ln(parent_node_visit)
        # returns -inf. We just return that value, disregarding all other values
        if parent:
            parent_node_visits = parent.visits
        else:
            return -inf
        #if DEBUG: print("Printing wins: {}".format(current.wins))
        #if DEBUG: print("Printing visists: {}".format(current.visits)

        first_term = 0.0
        if identitity == THIS_IDENTITIY:
            first_term = current.wins / float(current.visits)
        elif identitity != THIS_IDENTITIY:
            first_term = 1 - current.wins / float(current.visits)
        part =  sqrt(log(parent_node_visits) / float(current.visits))
        second_term = explore_faction * part

        #if DEBUG: print("The first: {}, second {}".format(first_term, second_term))
        return first_term + second_term

    #move, value
    active_node = node;
    active_node.visits += 1

    children = active_node.child_nodes
    active_uct_val = UCT(active_node.parent, active_node, identity)

    UCT_to_nodes = {}
    UCT_to_nodes[active_node] = active_uct_val

    last_node = None
    while children:
        # If a non leaf node is selected, by definition it has children
        # which means we need to externally break the loop.
        # We check this by seeing if the same node is selected twice. If a non
        # leaf node is selected, it will go into competition with the same set
        # once again, and will win, once again. Thus the next if statement
        # breaking out of the loop.
        #  Otherwise the leaf node is selected and it has no children.
        if last_node == active_node:
            break
        last_node = active_node

        #child move_to_child is a move
        for move_to_child in children:
            child = active_node.child_nodes[move_to_child]
            child.visits += 1

            UCT_value = UCT(active_node, child, identity)
            UCT_to_nodes[child] = UCT_value

        #The node with the max value
        #So here if it is not a leaf node, it is going to go into competition
        #with it's same set of children which will break out of the loop
        active_node = max(UCT_to_nodes, key=UCT_to_nodes.get)
        children = active_node.child_nodes

        if DEBUG: print(active_node)
    return active_node

    # Hint: return leaf_node

#adding a new MCTSNode to the tree
def expand_leaf(parent_node: MCTSNode, state, board):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        state:  The state of the game.

    Returns:    The added child node.

    """
    # The parent node can execute all the actions in the current state
    current_state = ravel_states(board, state, parent_node)
    parent_node.untried_actions = board.legal_actions(current_state)
    if len(parent_node.untried_actions) == 0:
        print("Cant expand leaf there are no possıble plays proceed.")
        parent_node.parent.wins = -inf
        parent_node.wins = -inf
        return None

    #select a random action that can be executed in that node
    #!!! Make this random. It is kinda random?
    p_action = parent_node.untried_actions.pop()

    #!!! action list might not be correct.
    #create a new node which would be the next state as a result of the chosen action
    new_node = MCTSNode(parent=parent_node, parent_action = p_action, action_list= parent_node.untried_actions)

    parent_node.child_nodes[p_action] = new_node
    return new_node
    # Hint: return new_node

def rollout_handler(worker_stack, move_count):
        with closing( Pool(move_count) ) as pools:
            return (pools.starmap(rollout_worker, worker_stack))

def rollout_worker(ROLLOUTS, MAX_DEPTH, board, state, me, move):
    total_score = 0.0
    # Sample a set number of games where the target move is immediately applied.
    for r in range(ROLLOUTS):
        rollout_state = board.next_state(state, move)

        # Only play to the specified depth.
        for i in range(MAX_DEPTH):
            if board.is_ended(rollout_state):
                break

            rollout_move = heuristic(rollout_state, board, me)

            if not rollout_move:
                continue

            rollout_state = board.next_state(rollout_state, rollout_move)
        outcome_result =outcome(board.owned_boxes(rollout_state),
                               board.points_values(rollout_state), me)
        total_score += outcome_result

    expectation = float(total_score) / float(ROLLOUTS)

    return (move, expectation)


# Returns choice a choice
def heuristic(state, board, identitity):
        legal_actions = board.legal_actions(state)

        # Seperate legal action from the list that it is going to loop
        # on as it sometimes creates weird behavior to modify the list you are
        # going through, even though it SHOULD be fine in this case.

        legal_actions_loop = copy.copy(legal_actions)
        #iterate through all possible actions in the state
        for action in legal_actions_loop:
            next_state = board.next_state(state, action)


            ### HEURISTIC ONE - CHECK THE INCREASE IN OWNED BOXES ###
            current_boxes = board.owned_boxes(state)
            current_box_owner_count = 0
            current_box_enemy_count = 0
            for value in current_boxes.values():
                if value == THIS_IDENTITIY:
                    current_box_owner_count += 1
                if value == OTHER_IDENTITIY:
                    current_box_enemy_count += 1

            next_boxes = board.owned_boxes(next_state)
            next_box_owner_count = 0
            next_box_enemy_count = 0
            for value in next_boxes.values():
                if value == THIS_IDENTITIY:
                    next_box_owner_count += 1
                if value == OTHER_IDENTITIY:
                    next_box_enemy_count += 1

            # if an action results in an main box increase, do that
            if next_box_owner_count > current_box_owner_count:
                return action

            # if an action results in an increase in enemy box count
            # remove that from the possible list of action
            if next_box_enemy_count > current_box_enemy_count:
                if action in legal_actions: legal_actions.remove(action)

            ### -------------------------------------------------------- ###

            ### HEURISTIC 2 - CHECK IF THE MOVES GIVES YOU THE WIN/LOSS ###
            final_dict = board.points_values(next_state)
            #Sometimes final_dict returns None. This behavior is outside the documentation
            #and I couldnt figure out when it is the case, so the next line is a
            #safe guard against that
            if final_dict:
                #For the player
                if identitity == THIS_IDENTITIY:
                    # if this move brings us to victory return it
                    if final_dict[THIS_IDENTITIY] == 1:
                        return action;

                    # if this move brings us defeat remove it from the possible actions
                    if final_dict[THIS_IDENTITIY] == -1:
                        if action in legal_actions: legal_actions.remove(action)
                #For the opponent
                else:
                    # If this move brings victory to the opponent remove it from possible actions
                    if final_dict[OTHER_IDENTITIY] == 1:
                        if action in legal_actions: legal_actions.remove(action)
                    # If this move brings defeat to the opponent return it
                    if final_dict[OTHER_IDENTITIY] == -1:
                        return action;
            ### ----------------------------------------------------------------- ###
        try:
            #if a terminal state isnt found just return a random legal action.
            return choice(legal_actions)
        except IndexError:
            print("Returned none from heuristic.")
            return None

def outcome(owned_boxes, game_points, me):
    if game_points is not None:
        # Try to normalize it up?  Not so sure about this code anyhow.
        red_score = game_points[THIS_IDENTITIY]*9
        blue_score = game_points[OTHER_IDENTITIY]*9
    else:
        red_score = len([v for v in owned_boxes.values() if v == THIS_IDENTITIY])
        blue_score = len([v for v in owned_boxes.values() if v == OTHER_IDENTITIY])
    return red_score - blue_score if me == THIS_IDENTITIY else blue_score - red_score

def rollout(node:MCTSNode, state, board):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        state:  The state of the game.

    """
    state = ravel_states(board, state, node)
    ROLLOUTS = 10
    MAX_DEPTH = 5
    moves = board.legal_actions(state)

    #Safe guard into not having enough plays.
    if len(moves) == 0:
        standing = board.points_values(state)
        print("Cant rollout as there are no possıble plays proceed.")
        if standing[THIS_IDENTITIY] == 0:
            return node.parent_action, 0
        elif standing[THIS_IDENTITIY] == 1:
            return node.parent_action, inf
        elif standing[THIS_IDENTITIY] == -1:
            return node.parent_action, -inf

    best_move = moves[0]
    best_expectation = float('-inf')

    me = board.current_player(state)

    worker_individual = [ROLLOUTS, MAX_DEPTH, board, state, me]
    worker_stack = [worker_individual + [move] for move in moves]

    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    # Num threads
    expectations = rollout_handler(worker_stack, NUM_THREADS)

    expectations.sort(key=lambda x: x[1])

    best_move, best_expectation = expectations[-1][0], expectations[-1][1]
    #print("Vaniilla bot picking %s with expected score %f" % (str(best_move), best_expectation))
    return best_move, best_expectation



#update all nodes along the path visited
def backpropagate(added_node: MCTSNode, expectation):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    added_node.wins = expectation
    while added_node.parent:
        if added_node.parent.wins < expectation:
            added_node.parent.wins = expectation
        added_node = added_node.parent


def think(board, state):
    global THIS_IDENTITIY
    global OTHER_IDENTITIY
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    if THIS_IDENTITIY == 0:
        THIS_IDENTITIY = identity_of_bot


    if THIS_IDENTITIY == 1: OTHER_IDENTITIY = 2
    elif THIS_IDENTITIY == 2: OTHER_IDENTITIY = 1

    if DEBUG: print("{} is the parralel bot".format(THIS_IDENTITIY))

    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    best_move = (0,0,0,0);
    best_expectation = -inf;

    if not TIME_CONSTRAINT:
        #Using the steps (num nodes)
        for step in range(num_nodes):
            # Copy the game for sampling a playthrough
            sampled_game = state
            # Start at root
            node = root_node

            # Do MCTS - This is all you!
            leaf_node = traverse_nodes(node, sampled_game, identity_of_bot)
            added_node = expand_leaf(leaf_node, sampled_game, board)

            # Failsafe in case added node has no possible plays
            while not added_node:
                leaf_node = traverse_nodes(node, sampled_game, identity_of_bot)
                added_node = expand_leaf(leaf_node, sampled_game, board)

            active_move, active_expectation = rollout(added_node, sampled_game, board)
            backpropagate(added_node, active_expectation)

            if active_expectation > best_expectation:
                best_move = active_move
                best_expectation = active_expectation
    else:
        # Using the timer!
        start = time.time()
        while time.time() - start < max_time:
            # Copy the game for sampling a playthrough
            sampled_game = state
            # Start at root
            node = root_node

            # Do MCTS - This is all you!
            leaf_node = traverse_nodes(node, sampled_game, identity_of_bot)
            added_node = expand_leaf(leaf_node, sampled_game, board)

            # Failsafe in case added node has no possible plays
            while not added_node:
                leaf_node = traverse_nodes(node, sampled_game, identity_of_bot)
                added_node = expand_leaf(leaf_node, sampled_game, board)

            active_move, active_expectation = rollout(added_node, sampled_game, board)
            backpropagate(added_node, active_expectation)

            if active_expectation > best_expectation:
                best_move = active_move
                best_expectation = active_expectation


    print("Paralel Heuristic:" + str(THIS_IDENTITIY) + " |Printing best move for paralel:" + str(best_move) + " with the expectation: " + str(best_expectation))
    #best_move = find_root_move(best_move, best_move_node)
    return best_move

