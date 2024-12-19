from game import TTT3D
from deep_q_learning import train_dql_agent, DQLAgent, dql_player
from tourney import ab_player
from mcts_hybrid import hybrid_player
import torch

def train_new_agent():
    """Train a new DQL agent and save it"""
    print("Starting training...")
    trained_agent = train_dql_agent(num_episodes=1000)  
    
    torch.save(trained_agent.q_network.state_dict(), "ttt3d_dql_model.pth")
    print("Training completed and model saved!")
    return trained_agent

def run_tournament(num_games=10):
    """Run a tournament between different players"""
    game = TTT3D()
    
    players = {
        "DQL": dql_player,
        "Alpha-Beta-1": ab_player(1),
        "Alpha-Beta-2": ab_player(2),
        "MCTS-Hybrid": hybrid_player
    }
    
    results = {name: {"wins": 0, "losses": 0, "draws": 0} for name in players}
    
    for p1_name, p1 in players.items():
        for p2_name, p2 in players.items():
            if p1_name >= p2_name: 
                continue
                
            print(f"\nPlaying {num_games} games: {p1_name} vs {p2_name}")
            
            for game_num in range(num_games):
                print(f"Game {game_num + 1}...")
                
                result = game.play_game(p1, p2)
                
                if result == 1:  # p1 wins
                    results[p1_name]["wins"] += 1
                    results[p2_name]["losses"] += 1
                elif result == -1:  # p2 wins
                    results[p1_name]["losses"] += 1
                    results[p2_name]["wins"] += 1
                else:  # draw
                    results[p1_name]["draws"] += 1
                    results[p2_name]["draws"] += 1
                
                print(f"Result: {'P1 wins' if result == 1 else 'P2 wins' if result == -1 else 'Draw'}")
    
    print("\nTournament Results:")
    print("=" * 50)
    for player, score in results.items():
        total_games = score["wins"] + score["losses"] + score["draws"]
        win_rate = (score["wins"] + 0.5 * score["draws"]) / total_games if total_games > 0 else 0
        print(f"\n{player}:")
        print(f"Wins: {score['wins']}")
        print(f"Losses: {score['losses']}")
        print(f"Draws: {score['draws']}")
        print(f"Win Rate: {win_rate:.2%}")

if __name__ == "__main__":
    print("Do you want to train a new agent? (y/n)")
    if input().lower() == 'y':
        trained_agent = train_new_agent()
    
    # Run the tournament
    print("\nDo you want to run a tournament? (y/n)")
    if input().lower() == 'y':
        num_games = int(input("How many games per matchup? "))
        run_tournament(num_games)