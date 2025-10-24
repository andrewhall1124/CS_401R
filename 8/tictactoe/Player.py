# Player.py
import random
from TicTacToe import TicTacToe
import numpy as np
from rich import print


class Player:
    def __init__(self, letter):
        # letter is 'X' or 'O'
        self.letter = letter
        self.opponent_letter = 'O' if self.letter == 'X' else 'X'

    def get_move(self, game: TicTacToe):
        pass


class RandomPlayer(Player):
    def get_move(self, game: TicTacToe):
        # Randomly choose a valid move
        square = random.choice(game.empty_squares())
        return square


class HumanPlayer(Player):
    def get_move(self, game: TicTacToe):
        # Ask user for input
        valid_square = False
        val = None
        while not valid_square:
            square = input(f"Your turn ({self.letter}). Input move (0-8): ")
            try:
                val = int(square)
                if val not in game.empty_squares():
                    raise ValueError
                valid_square = True
            except ValueError:
                print("Invalid move. Try again.")
        return val


class OptPlayer(Player):

    def __init__(self, letter):
        super().__init__(letter)
        print('finding optimal policy...')
        game = TicTacToe()
        self.policy = {}
        self.explore('X', game)
        print('done')

    def explore(self, player: Player, game: TicTacToe):
        res = game.result()
        if res == 'not done':
            best_move = ''
            best_score = -float('inf') if player == self.letter else float('inf')
            for move in game.empty_squares():
                new_game = TicTacToe()
                new_game.board = game.board.copy()
                new_game.make_move(move, player)
                if player == self.letter:
                    new_player = self.opponent_letter
                else:
                    new_player = self.letter
                score = self.explore(new_player, new_game)
                if player == self.letter and score > best_score:
                    best_score = score
                    best_move = move
                elif player == self.opponent_letter and score < best_score:
                    best_score = score
                    best_move = move
            if player == self.letter:
                state = tuple(game.board)
                self.policy[state] = best_move
            return best_score

        elif res == self.letter:
            return 1
        elif res == self.opponent_letter:
            return -1
        else:
            return 0

    def get_move(self, game):
        state = tuple(game.board)
        return self.policy[state]


class RLPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.V = {}                      # State-value function
        self.policy = {}                 # Policy mapping state to action
        self.states = []

    ############################################################################
    # this is the function you will implement policy iteration with Monte-Carlo
    # simulation for the policy evaluation step
    ##########################################################
    def train(self, N = 1000, gamma = 0.9, max_iterations = 100):
        print("Training RLPlayer!")

        # Initialize policy
        game = TicTacToe()
        self.generate_all_states(game, self.letter)
        for state in self.states:
            self.V[state] = 0.0
            game_temp = TicTacToe()
            game_temp.board = list(state)
            available = game_temp.empty_squares()
            if len(available) > 0:
                self.policy[state] = random.choice(available)

        for i in range(max_iterations):
            # Evaluate policy
            self.policy_evaluation(N=N, gamma=gamma)

            # Improve policy
            new_policy = self.policy_improvement()

            # Check for stability
            stable = True
            for state in new_policy.keys():
                if state in self.policy and self.policy[state] != new_policy[state]:
                    stable = False
                    break

            # Update policy with new policy
            self.policy.update(new_policy)

            if stable:
                print(f"Converged at iteration {i}!")
                return

        print(f"Max iterations ({max_iterations}) reached without convergence.")


    def policy_evaluation(self, N: int, gamma: float) -> None:
        returns = {state: [] for state in self.states}

        for _ in range(N):
            game = TicTacToe()
            current_player = 'X'
            state_sequences = []

            while game.result() == 'not done':
                state = self.get_state(game)

                # Our turn
                if current_player == self.letter:
                    move = self.policy.get(state, random.choice(game.empty_squares()))
                    state_sequences.append(state)
                # Opponent turn (random move)
                else:
                    move = random.choice(game.empty_squares())

                game.make_move(move, current_player)
                current_player = 'O' if current_player == 'X' else 'X'

            result = game.result()
            if result == self.letter:
                reward = 1.0
            elif result == self.opponent_letter:
                reward = -1.0
            else:
                reward = 0.0

            # Calculate discounted return for each visited state
            G = reward
            for i in range(len(state_sequences) - 1, -1, -1):
                state = state_sequences[i]
                returns[state].append(G)
                # Discount for earlier states (2 steps back since opponent moves in between)
                G = gamma * gamma * G

        for state in self.states:
            # Rewards found
            if len(returns[state]) > 0:
                self.V[state] = np.mean(returns[state])
            # No rewards found
            else:
                self.V[state] = 0.0

    def policy_improvement(self) -> None:
        policy = {}
        for state in self.states:

            game = TicTacToe()
            game.board = list(state)
            available_moves = game.empty_squares()

            if not available_moves:
                continue

            best_action = None
            best_value = -float('inf')

            for action in available_moves:
                game_copy = TicTacToe()
                game_copy.board = list(state)
                game_copy.make_move(action, self.letter)
                next_state = self.get_state(game_copy)

                value = self.V.get(next_state, 0.0)

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action
        
        return policy


    def get_state(self, game: TicTacToe):
        # Convert the board to a tuple (immutable and hashable)
        return tuple(game.board)
    
    def get_move(self, game: TicTacToe):
        state = self.get_state(game)
        available_moves = game.empty_squares()
        # Ensure the policy has an action for the current state
        if state not in self.policy:
            self.policy[state] = random.choice(available_moves)
        return self.policy[state]
        
    def generate_all_states(self, game: TicTacToe, player_turn: str):
        # Generate all possible states where it's RLPlayer's turn
        state = self.get_state(game)
        # Only add the state if it's RLPlayer's turn
        if player_turn == self.letter and state not in self.states:
            self.states.append(state)
        # Check for terminal state
        if game.current_winner or game.is_full():
            return
        for action in game.empty_squares():
            # Make a copy of the game
            game_copy = TicTacToe()
            game_copy.board = game.board.copy()
            game_copy.current_winner = game.current_winner
            # Make a move
            game_copy.make_move(action, player_turn)
            # Switch player turn
            next_player_turn = 'O' if player_turn == 'X' else 'X'
            # Recursively generate states
            self.generate_all_states(game_copy, next_player_turn)

if __name__ == '__main__':
    rl_player = RLPlayer("X")
    rl_player.train()