from mcts_node import MCTSNode
from random import choice, random
from math import sqrt, log, inf


DEBUG = False

num_nodes = 10
explore_faction = 2
THIS_IDENTITIY = 0


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

    # Bad hint.
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

def rollout(node:MCTSNode, state, board):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        state:  The state of the game.

    """

    # Returns choice and corresponding heuristic weight
    def heuristic(state, board, identitity):
        weight = 0
        legal_actions = board.legal_actions(state)
        #iterate through all possible actions in the state
        for action in legal_actions:
            next_state = board.next_state(state, action)
            final_dict = board.points_values(next_state)
            #Sometimes this returns None. This behavior is outside the documentation
            #and I couldnt figure out when it is the case, so the next line is a
            #safe guard against that
            if final_dict:
                if DEBUG: print("Actual heuristic")
                if identitity == THIS_IDENTITIY:
                    if final_dict[THIS_IDENTITIY] == 1:
                        return action;
                    if final_dict[THIS_IDENTITIY] == -1:
                        if len(final_dict):
                            legal_actions.remove(action)
                else:
                    if final_dict[THIS_IDENTITIY] == 1:
                        if len(final_dict):
                            legal_actions.remove(action)
                        return action;
        try:
            return choice(legal_actions)
        except IndexError:
            print("Index error")
            return None

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

    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    def outcome(owned_boxes, game_points):
        if game_points is not None:
            # Try to normalize it up?  Not so sure about this code anyhow.
            red_score = game_points[1]*9
            blue_score = game_points[2]*9
        else:
            red_score = len([v for v in owned_boxes.values() if v == 1])
            blue_score = len([v for v in owned_boxes.values() if v == 2])
        return red_score - blue_score if me == THIS_IDENTITIY else blue_score - red_score

    for move in moves:
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
    global THIS_IDENTITIY
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    if THIS_IDENTITIY == 0:
        THIS_IDENTITIY = identity_of_bot
    print("{} is the heuristic bot".format(THIS_IDENTITIY))

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
            print("Failsafe activated.")
            leaf_node = traverse_nodes(node, sampled_game, identity_of_bot)
            added_node = expand_leaf(leaf_node, sampled_game, board)

        active_move, active_expectation = rollout(added_node, sampled_game, board)
        backpropagate(added_node, active_expectation)

        if active_expectation > best_expectation:
            best_move_node = added_node
            best_move = active_move
            best_expectation = active_expectation

    print("#####################Printing best move for heruistic:" + str(best_move) + "with the expectation: " + str(best_expectation))
    #best_move = find_root_move(best_move, best_move_node)
    return best_move

