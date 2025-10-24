# Player.py
import random
from TicTacToe import TicTacToe
import numpy as np
from rich import print

random.seed(42)


class Player:
    def __init__(self, letter):
        # letter is 'X' or 'O'
        self.letter = letter
        self.opponent_letter = "O" if self.letter == "X" else "X"

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
        print("finding optimal policy...")
        game = TicTacToe()
        self.policy = {}
        self.explore("X", game)
        print("done")

    def explore(self, player: Player, game: TicTacToe):
        res = game.result()
        if res == "not done":
            best_move = ""
            best_score = -float("inf") if player == self.letter else float("inf")
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
        self.V = {}  # State-value function
        self.policy = {}  # Policy mapping state to action
        self.states = []
        self.gamma = 0

    ############################################################################
    # this is the function you will implement policy iteration with Monte-Carlo
    # simulation for the policy evaluation step
    ##########################################################
    def train(
        self,
        eta_type="standard",
        N: int = 500,
        gamma: float = 0.1,
        epsilon: float = 0.001,
        opponent_policy: str = "random",
        max_iterations=100,
        seed: int = 42,
    ):
        print("Training RLPlayer!")
        self.gamma = gamma

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize all possible states and random policy
        self._initialize_policy()

        # Policy iteration loop
        for iteration in range(max_iterations):
            # Store previous values
            V_old = self.V.copy()

            # Evaluate current policy using Monte Carlo
            self.policy_evaluation(N)

            # Improve policy greedily
            self._improve_policy()

            # Compute infinity norm of value function change
            max_diff = max(abs(self.V[state] - V_old[state]) for state in self.states)

            if max_diff <= epsilon:
                print(f"Policy converged at iteration {iteration}! (max_diff: {max_diff:.6f})")
                return

        print(f"Max iterations ({max_iterations}) reached without convergence.")

    def _initialize_policy(self):
        game = TicTacToe()
        self._generate_all_states(game, "X")

        for state in self.states:
            self.V[state] = 0.0
            game.board = list(state)
            available = game.empty_squares()
            if available:
                self.policy[state] = random.choice(available)

    def policy_evaluation(self, N: int) -> None:
        returns = {state: [] for state in self.states}

        # Run N episodes
        for _ in range(N):
            episode_states, reward = self._simulate_episode()

            # Discount rewards (backwards)
            G = reward
            for state in reversed(episode_states):
                returns[state].append(G)
                G *= self.gamma * self.gamma

        # Update value function with average returns
        for state in self.states:
            if returns[state]:
                self.V[state] = np.mean(returns[state])
            else:
                self.V[state] = 0.0

    def _simulate_episode(self):
        game = TicTacToe()
        current_player = "X"
        visited_states = []

        while game.result() == "not done":
            state = tuple(game.board)

            if current_player == self.letter:
                # Follow policy for our moves
                move = self.policy.get(state)
                visited_states.append(state)
            else:
                # Opponent plays randomly
                move = random.choice(game.empty_squares())

            game.make_move(move, current_player)
            current_player = "O" if current_player == "X" else "X"

        # Compute final reward
        result = game.result()
        if result == self.letter:
            reward = 1.0
        elif result == self.opponent_letter:
            reward = -1.0
        else:
            reward = 0.0

        return visited_states, reward

    def _improve_policy(self) -> None:
        for state in self.states:
            game = TicTacToe()
            game.board = list(state)
            available_moves = game.empty_squares()

            if not available_moves:
                continue

            # Find action with highest expected value
            best_action, best_value = self._get_best_action(game, available_moves)

            # Update policy
            self.policy[state] = best_action

    def _get_best_action(self, game: TicTacToe, available_moves: list):
        best_action = None
        best_value = -float("inf")

        # Sort moves for deterministic tie-breaking
        for action in available_moves:
            value = self._compute_action_value(game, action)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def _compute_action_value(self, game: TicTacToe, action: int) -> float:
        temp_game = TicTacToe()
        temp_game.board = game.board.copy()
        temp_game.make_move(action, self.letter)

        # Check for immediate terminal state
        result = temp_game.result()
        if result == self.letter:
            return 1.0
        elif result == "T":
            return 0.0
        elif result == self.opponent_letter:
            return -1.0

        return self._compute_opponent_expectation(temp_game)

    def _compute_opponent_expectation(self, game: TicTacToe) -> float:
        opponent_moves = game.empty_squares()
        if not opponent_moves:
            return 0.0

        total_value = 0.0
        for opp_move in opponent_moves:
            # Simulate opponent move
            temp_game = TicTacToe()
            temp_game.board = game.board.copy()
            temp_game.make_move(opp_move, self.opponent_letter)

            # Check result after opponent move
            opp_result = temp_game.result()
            if opp_result == self.opponent_letter:
                total_value -= 1.0
            elif opp_result == "T":
                total_value += 0.0
            else:
                # Use value function for next state
                next_state = tuple(temp_game.board)
                total_value += self.V.get(next_state, 0.0)

        return total_value / len(opponent_moves)

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

    def _generate_all_states(self, game: TicTacToe, player_turn: str):
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
            next_player_turn = "O" if player_turn == "X" else "X"
            # Recursively generate states
            self._generate_all_states(game_copy, next_player_turn)


if __name__ == "__main__":
    rl_player = RLPlayer("X")
    rl_player.train(seed=42)
