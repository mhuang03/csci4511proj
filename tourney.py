import json
import sys

from games4e import monte_carlo_tree_search, alpha_beta_cutoff_search, random_player
from game import *
from mm_eval import mm_eval_good, mm_eval_funny
from mcts_hybrid_rollout import mcts_hybrid_rollout_search
from deep_q_learning import dql_player


def ab_player(depth, eval_fn=mm_eval_good):
    def player(game, state):
        return alpha_beta_cutoff_search(state, game, d=depth, eval_fn=eval_fn)
    return player


def mcts_player(game, state):
    return monte_carlo_tree_search(state, game, N=1000)


def mcts_hybrid_player(depth):
    def player(game, state):
        return mcts_hybrid_rollout_search(state, game, N=100, minimax_depth=depth)
    return player


if __name__ == "__main__":
    deterministic = {
        "AB1": ab_player(1),
        "AB2": ab_player(2),
        "Basic1": ab_player(1, eval_fn=None),
        "Basic2": ab_player(2, eval_fn=None),
        "Funny": ab_player(0, eval_fn=mm_eval_funny),
        "DQL": dql_player,
    }
    stochastic = {
        "Random": random_player,
        "MCTS": mcts_player,
        "MCTS-H1": mcts_hybrid_player(1),
    }
    all_players = [i for i in deterministic.items()] + [i for i in stochastic.items()]
    player1_list = all_players


    filename = "results/tourney_results.json"
    if sys.argv:
        if len(sys.argv) == 2:
            player = sys.argv[1]
            print(player)
            if player in deterministic:
                player1_list = [(player, deterministic[player])]
            elif player in stochastic:
                player1_list = [(player, stochastic[player])]
            else:
                print(f"Unknown player {player}")
                sys.exit(1)
        elif len(sys.argv) == 3:
            all_players_dict = {k: v for k, v in all_players}
            player1 = sys.argv[1]
            player2 = sys.argv[2]

            if player1 not in all_players_dict or player2 not in all_players_dict:
                print(f"Unknown player {player1} or {player2}")
                sys.exit(1)

            tt = TTT3D()
            result = tt.play_game(all_players_dict[player1], all_players_dict[player2], verbose=True)
            sys.exit(0)


    try:
        for player1 in player1_list:
            p1_name, p1 = player1
            filename = f"results/tourney_results_{p1_name}.json"

            results = {}
            try:
                with open(filename, "r") as file:
                    results = json.load(file)
            except Exception as _:
                pass

            for player2 in all_players:
                p2_name, p2 = player2

                x_wins, o_wins, ties = 0, 0, 0
                first, times = 0, 100

                if p1_name in deterministic and p2_name in deterministic:
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

            with open(filename, "w") as file:
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
        with open(filename, "w") as file:
            json.dump(results, file, indent=4)