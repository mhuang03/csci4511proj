import numpy as np
from typing import Optional, Tuple
from games4e import Game, GameState

class HybridNode:
    """Node for hybrid MCTS-Minimax search."""
    def __init__(self, state: GameState, parent: Optional['HybridNode'] = None):
        self.state = state
        self.parent = parent
        self.children: dict['HybridNode', tuple] = {}  # Child nodes and their moves
        self.visits = 0
        self.value = 0.0
        self.minimax_value: Optional[float] = None
        
def hybrid_search(game: Game, state: GameState, num_simulations: int = 1000, 
                 minimax_depth: int = 2) -> Tuple[int, int, int]:
    """
    Hybrid MCTS-Minimax search algorithm.
    Returns the best move found.
    """
    root = HybridNode(state)
    
    def select(node: HybridNode) -> HybridNode:
        """Select a leaf node using UCB1 and minimax values."""
        while node.children and not game.terminal_test(node.state):
            if not all(child.minimax_value is not None for child in node.children):
                node = max(node.children.keys(), 
                         key=lambda n: ucb_with_minimax(n, game.to_move(state)))
            else:
                node = max(node.children.keys(),
                         key=lambda n: n.minimax_value if game.to_move(state) == 1 
                                     else -n.minimax_value)
        return node
    
    def expand(node: HybridNode) -> HybridNode:
        """Expand node and perform shallow minimax evaluation."""
        if not node.children and not game.terminal_test(node.state):
            for move in game.actions(node.state):
                child_state = game.result(node.state, move)
                child = HybridNode(child_state, parent=node)
                node.children[child] = move
                
                child.minimax_value = minimax(game, child_state, minimax_depth)
        return select(node)
    
    def simulate(node: HybridNode) -> float:
        """Run a simulation from node."""
        state = node.state
        while not game.terminal_test(state):
            move = np.random.choice(game.actions(state))
            state = game.result(state, move)
        return game.utility(state, game.to_move(root.state))
    
    def backpropagate(node: HybridNode, value: float) -> None:
        """Update node statistics."""
        while node is not None:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent
            
    def ucb_with_minimax(node: HybridNode, player: int, C: float = 1.4) -> float:
        """UCB1 formula incorporating minimax values when available."""
        if node.visits == 0:
            return float('inf')
        
        ucb = node.value / node.visits + C * np.sqrt(np.log(node.parent.visits) / node.visits)
        
        if node.minimax_value is not None:
            minimax_component = node.minimax_value if player == 1 else -node.minimax_value
            ucb = 0.7 * ucb + 0.3 * minimax_component
            
        return ucb
    
    def minimax(game: Game, state: GameState, depth: int) -> float:
        """Minimax evaluation to limited depth."""
        if depth == 0 or game.terminal_test(state):
            return evaluate_position(state)
            
        values = []
        for move in game.actions(state):
            next_state = game.result(state, move)
            values.append(minimax(game, next_state, depth - 1))
            
        return max(values) if game.to_move(state) == 1 else min(values)
    
    def evaluate_position(state: GameState) -> float:
        """Simple position evaluation based on piece configurations."""
        board = state.board
        score = 0
        
        for i in range(4):
            for j in range(4):
                line = [board[i][j][k] for k in range(4)]
                score += evaluate_line(line)
                
                line = [board[i][k][j] for k in range(4)]
                score += evaluate_line(line)
                
                line = [board[k][i][j] for k in range(4)]
                score += evaluate_line(line)
        
        for i in range(4):
            line = [board[i][j][j] for j in range(4)]
            score += evaluate_line(line)
            line = [board[j][i][j] for j in range(4)]
            score += evaluate_line(line)
            line = [board[j][j][i] for j in range(4)]
            score += evaluate_line(line)
            
        return score
    
    def evaluate_line(line: list) -> float:
        """Evaluate a single line of pieces."""
        player_count = sum(1 for x in line if x == 1)
        opponent_count = sum(1 for x in line if x == -1)
        
        if player_count > 0 and opponent_count > 0:
            return 0  
        elif player_count > 0:
            return 10 ** (player_count - 1)
        elif opponent_count > 0:
            return -(10 ** (opponent_count - 1))
        return 0
    
    for _ in range(num_simulations):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child)
        backpropagate(child, result)
    
    best_child = max(root.children.keys(), key=lambda n: n.visits)
    return root.children[best_child]

def hybrid_player(game, state):
    """Player function that uses hybrid search."""
    return hybrid_search(game, state)