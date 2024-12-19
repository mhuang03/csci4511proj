import json

from games4e import monte_carlo_tree_search, alpha_beta_cutoff_search, random_player
from game import *
from mm_eval import mm_eval
from mcts_hybrid_rollout import mcts_hybrid_rollout_search


def mcts_player(game, state):
    return monte_carlo_tree_search(state, game, N=1000)


def ab_player(depth):
    def player(game, state):
        return alpha_beta_cutoff_search(state, game, d=depth, eval_fn=mm_eval_good)
    return player


def mcts_hybrid_player(depth):
    def player(game, state):
        return mcts_hybrid_rollout_search(state, game, N=100, minimax_depth=depth)
    return player


if __name__ == "__main__":
    deterministic = [
        (ab_player(1), "AB1"),
        (ab_player(2), "AB2"),
    ]
    stochastic = [
        (random_player, "Random"),
        (mcts_player, "MCTS"),
        (mcts_hybrid_player(1), "MCTS-H1"),
    ]
    all_players = deterministic + stochastic

    results = {}
    try:
        with open("tourney_results.json", "r") as file:
            results = json.load(file)
    except Exception as _:
        pass

    try:
        for player1 in all_players:
            for player2 in all_players:
                p1, p1_name = player1
                p2, p2_name = player2

                x_wins, o_wins, ties = 0, 0, 0
                first, times = 0, 100

                if player1 in deterministic and player2 in deterministic:
                    times = 1

                if f"{p1_name} vs {p2_name}" in results:
                    if results[f"{p1_name} vs {p2_name}"]["done"]:
                        continue
                    else:
                        x_wins = results[f"{p1_name} vs {p2_name}"]["X"]
                        o_wins = results[f"{p1_name} vs {p2_name}"]["O"]
                        ties = results[f"{p1_name} vs {p2_name}"]["ties"]
                        first = x_wins + o_wins + ties


                for i in range(first, times):
                    tt = TTT3D()
                    result = tt.play_game(p1, p2)
                    if result == 1:
                        x_wins += 1
                    elif result == -1:
                        o_wins += 1
                    else:
                        ties += 1
                    print(f"{p1_name} vs {p2_name} game {i}: {result}")

                results[f"{p1_name} vs {p2_name}"] = {
                    "X": x_wins,
                    "O": o_wins,
                    "ties": ties,
                    "done": True
                }
                print(f"X: {p1_name}, O: {p2_name}; X wins {x_wins} times, O wins {o_wins} times, {ties} ties")

        with open("tourney_results.json", "w") as file:
            json.dump(results, file, indent=4)

    except KeyboardInterrupt:
        print("Interrupted")
        results[f"{p1_name} vs {p2_name}"] = {
            "X": x_wins,
            "O": o_wins,
            "ties": ties,
            "done": False
        }

    finally:
        with open("tourney_results.json", "w") as file:
            json.dump(results, file, indent=4)