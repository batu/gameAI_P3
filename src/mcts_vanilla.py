from mcts_node import MCTSNode
from random import choice, random
from math import sqrt, log, inf


DEBUG = False
NON_LEAF_CHANCE = .8

num_nodes = 100
explore_faction = 5

#board.legal_actions(state) returns the moves available in state.
#board.next_state(state, action) returns a new state constructed by applying action in state.
#board.is_ended(state) returns True if the game has ended in state and False otherwise.
#board.current_player(state)  returns the index of the current player in state.
#board.points_values(state)  returns a dictionary of the score for each player (eg {1:-1,2:1} for a second-player win).  Will return {1: 0, 2: 0} if the game is not ended.
#board.owned_boxes(state)  returns a dict with (Row,Column) keys; values indicate for each box whether player 1, 2, or 0 (neither) owns that box
#board.display(state)  returns a string representation of the board state.
#board.display_action(action)  returns a string representation of the game action.

def find_root_move(best_move, node: MCTSNode):
    parent_move = best_move
    while node.parent:
        parent_move = node.parent_action
        node = node.parent
    return parent_move

def ravel_states(board, root_state, current_node: MCTSNode):
    moves = []
    parent_count = 0

    while current_node.parent:
        moves.append(current_node.parent_action)
        current_node = current_node.parent
        parent_count += 1
        if DEBUG:
            try:
                #print("At {} level there are {} nodes".format(parent_count, len(current_node.child_nodes)))
                pass
            except:
                print("what")
                pass
    if DEBUG: print("We were {} deep!".format(parent_count))
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

    #Returns a random leaf node.
    active_node = node;
    children = active_node.child_nodes
    visited_count = 0
    all_visited_this_level = False

    # currently always goes for a leaf node.

    while children:
        non_leaf_chance = random()
        #child key is a move

        for child_key in children:
            #one of the many children
            if active_node.visits <= visited_count:
                active_node = active_node.child_nodes[child_key]
                children = active_node.child_nodes
            #make sure you visited all the unexpended nodes before moving to the next level
            visited_count += 1

        if non_leaf_chance > NON_LEAF_CHANCE:
            break
    active_node.visits += 1
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
        print("Cant expand this leaf node.")
        return None


    #select a random action that can be executed in that node
    p_action = parent_node.untried_actions.pop()

    #create a new node which would be the next state as a result of the chosen action
    new_node = MCTSNode(parent=parent_node, parent_action = p_action, action_list= parent_node.untried_actions)

    parent_node.child_nodes[p_action] = new_node
    return new_node
    # Hint: return new_node

#simulating the remainder of the game
def rollout(node:MCTSNode, state, board):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        state:  The state of the game.

    """
    state = ravel_states(board, state, node)

    ROLLOUTS = 10
    MAX_DEPTH = explore_faction
    moves = board.legal_actions(state)

    if len(moves) == 0:
        standing = board.points_values(state)
        print("Cant rollout as there are no possıble plays proceed.")
        if standing[1] == 0:
            return node.parent_action, 0
        elif standing[1] == 1:
            return node.parent_action, inf
        elif standing[1] == -1:
            return node.parent_action, -inf

    best_move = moves[0]
    best_expectation = float('-inf')

    me = board.current_player(state)

    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    def outcome(owned_boxes, game_points):
        if game_points is not None:
            # Try to normalize it up?  Not so sure about this code anyhow.
            red_score = game_points[1]*9
            blue_score = game_points[2]*9
        else:
            red_score = len([v for v in owned_boxes.values() if v == 1])
            blue_score = len([v for v in owned_boxes.values() if v == 2])
        return red_score - blue_score if me == 1 else blue_score - red_score

    for move in moves:
        total_score = 0.0

        # Sample a set number of games where the target move is immediately applied.
        for r in range(ROLLOUTS):
            rollout_state = board.next_state(state, move)

            # Only play to the specified depth.
            for i in range(MAX_DEPTH):
                if board.is_ended(rollout_state):
                    break
                rollout_move = choice(board.legal_actions(rollout_state))
                rollout_state = board.next_state(rollout_state, rollout_move)

            total_score += outcome(board.owned_boxes(rollout_state),
                                   board.points_values(rollout_state))


        expectation = float(total_score) / ROLLOUTS

        # If the current move has a better average score, replace best_move and best_expectation
        if expectation > best_expectation:
            best_expectation = expectation
            best_move = move

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
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    best_move = (0,0,0,0);
    best_expectation = -inf;
    best_move_node = None
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
            best_move_node = added_node
            best_move = active_move
            best_expectation = active_expectation

    print("#####################Printing best move:" + str(best_move) + "with the expectation: " + str(best_expectation))
    #best_move = find_root_move(best_move, best_move_node)
    return best_move


    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    return None
