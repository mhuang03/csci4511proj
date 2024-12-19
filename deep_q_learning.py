import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple
from game import TTT3D
from games4e import GameState

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TTT3DNet(nn.Module):
    """Neural network for 3D Tic-Tac-Toe Q-learning."""
    def __init__(self):
        super(TTT3DNet, self).__init__()
        
        # Input: 4x4x4 board state (flattened to 64) + player turn
        self.input_size = 64 + 1
        
        # Three possible actions per position (empty, X, O)
        self.output_size = 64
        
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQLAgent:
    """Deep Q-Learning agent for 3D Tic-Tac-Toe."""
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (main and target)
        self.q_network = TTT3DNet().to(self.device)
        self.target_network = TTT3DNet().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Target network update frequency
        self.target_update = 10
        self.steps = 0
        
    def state_to_tensor(self, state: GameState) -> torch.Tensor:
        """Convert game state to tensor representation."""
        # Flatten the 3D board
        board_flat = [x for board in state.board for row in board for x in row]
        
        # Add player turn
        state_vector = board_flat + [1 if state.to_move == 1 else -1]
        
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def select_action(self, state: GameState, legal_moves: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
            
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            
            # Create mask for legal moves
            legal_moves_mask = torch.zeros(64, device=self.device)
            for move in legal_moves:
                idx = move[0] * 16 + move[1] * 4 + move[2]
                legal_moves_mask[idx] = 1
                
            # Set illegal moves to negative infinity
            q_values = q_values.squeeze()
            q_values[legal_moves_mask == 0] = float('-inf')
            
            # Select best legal move
            best_idx = q_values.argmax().item()
            return (best_idx // 16, (best_idx % 16) // 4, best_idx % 4)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step using replay buffer."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        experiences = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.cat([self.state_to_tensor(e.state) for e in experiences])
        actions = torch.tensor([(e.action[0] * 16 + e.action[1] * 4 + e.action[2]) 
                              for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], 
                             device=self.device, dtype=torch.float32)
        next_states = torch.cat([self.state_to_tensor(e.next_state) 
                               for e in experiences])
        dones = torch.tensor([e.done for e in experiences], 
                           device=self.device, dtype=torch.float32)
        
        # Compute Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def train_dql_agent(num_episodes: int = 1000):
    """Train DQL agent through self-play."""
    game = TTT3D()
    agent = DQLAgent()
    
    for episode in range(num_episodes):
        state = game.initial
        total_reward = 0
        moves_made = 0
        
        while not game.terminal_test(state):
            # Get legal moves
            legal_moves = game.actions(state)
            
            # Select and perform action
            action = agent.select_action(state, legal_moves)
            next_state = game.result(state, action)
            moves_made += 1
            
            # Calculate reward
            reward = 0
            if game.terminal_test(next_state):
                utility = game.utility(next_state, state.to_move)
                if utility > 0:
                    reward = 1.0  # Win
                elif utility < 0:
                    reward = -1.0  # Loss
                else:
                    reward = 0.0  # Draw
            else:
                reward = -0.01  # Small negative reward for non-terminal moves
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, 
                                 game.terminal_test(next_state))
            
            # Train
            loss = agent.train_step()
            
            total_reward += reward
            state = next_state
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Moves: {moves_made}, Epsilon: {agent.epsilon:.3f}")
    
    return agent

def dql_player(game, state):
    """Player function that uses trained DQL agent."""
    # Try to load pre-trained model, if it exists
    agent = DQLAgent()
    try:
        agent.q_network.load_state_dict(torch.load("ttt3d_dql_model.pth"))
        agent.epsilon = 0.0  # No exploration during actual play
    except FileNotFoundError:
        print("Warning: No pre-trained model found. Using untrained agent.")
    
    return agent.select_action(state, game.actions(state))

if __name__ == "__main__":
    # Example usage
    trained_agent = train_dql_agent(num_episodes=1000)
    torch.save(trained_agent.q_network.state_dict(), "ttt3d_dql_model.pth")