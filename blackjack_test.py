from IPython.display import clear_output
import random
import numpy as np
import matplotlib.pyplot as plt

DEALER_THRESHOLD = 16
BLACKJACK = 21

# ENVIRONMENT AND BLACKJACK LOGIC
def genDeck(shuffle=True):
    deck = np.array([[11 if x == 1 else 10 if x > 10 else x] * 4 for x in range(1, 14)] * 6)
    deck = deck.flatten()
    if shuffle:
        random.shuffle(deck)
    return list(deck)

def draw_card(deck):
    index = random.randint(0, len(deck) - 1)
    return deck.pop(index)

def sum_hand(hand):
    sum_value = sum(hand)
    num_aces = hand.count(11)

    while sum_value > BLACKJACK and num_aces:
        sum_value -= 10  # Convert an Ace from 11 to 1
        num_aces -= 1

    return sum_value

def dealer_logic(dealer_hand, deck):
    if sum_hand(dealer_hand) <= DEALER_THRESHOLD:
        card = draw_card(deck)
        print(f"Dealer drew: {card}")
        dealer_hand.append(card)
    else:
        print("Dealer stands")

def evaluate(player_stood, user_hand, dealer_hand):
    user_sum, dealer_sum = sum_hand(user_hand), sum_hand(dealer_hand)

    if dealer_sum > BLACKJACK:
        print(f"Dealer bust with: {dealer_sum, dealer_hand}")
        return True
    elif user_sum > BLACKJACK:
        print(f"Player bust with: {user_sum, user_hand}")
        return False
    elif dealer_sum == BLACKJACK:
        print(f"Dealer won with blackjack: {dealer_hand}")
        return False
    elif user_sum == BLACKJACK:
        print(f"Player won with blackjack: {user_hand}")
        return True
    elif dealer_sum > DEALER_THRESHOLD:
        if not player_stood:
            return None
        if dealer_sum > user_sum:
            print(f"Dealer won with: {dealer_sum, dealer_hand}")
            return False
        else:
            print(f"Player won with: {user_sum, user_hand}")
            return True
    else:
        return None

def play_round(deck):
    dealer_hand, user_hand = [draw_card(deck)], [draw_card(deck)]
    result, player_stood = None, False

    while result is None:
        print("Player's hand: ", user_hand)
        print("Dealer's hand: ", dealer_hand)

        while True:
            clear_output(wait=True)
            user_input = input("0 to stand, 1 to draw: ")
            if user_input.isdigit() and user_input in ['0', '1']:
                user_input = int(user_input)
                break
            else:
                print("Invalid input. Please enter 0 or 1.")

        if user_input == 1:
            card = draw_card(deck)
            print(f"Player drew: {card}")
            user_hand.append(card)
            player_stood = False

        if user_input == 0:
            print(f"Player stands with: {user_hand}")
            player_stood = True
            while result is None:
                dealer_logic(dealer_hand, deck)
                result = evaluate(player_stood, user_hand, dealer_hand)
            break
            

        dealer_logic(dealer_hand, deck)
        result = evaluate(player_stood, user_hand, dealer_hand)
    
    return result

def play(prev_state, action, deck):

    done = False
    reward = 0
    result = None
    
    dealer_hand, user_hand = prev_state
    if action == 1:
        card = draw_card(deck)
        user_hand.append(card)
        print(f"Player drew: {card}")
        dealer_logic(dealer_hand, deck)
        result = evaluate(False, user_hand, dealer_hand)
        if result is not None:
            done = True

    if action == 0:
        while result is None:
            dealer_logic(dealer_hand, deck)
            result = evaluate(True, user_hand, dealer_hand)
        done = True
    
    diff = np.abs(sum_hand(dealer_hand) - sum_hand(user_hand)) / 100

    if result == True:
        reward += 13 #+ diff
    elif result == False:
        reward -= 13 - diff
    elif result == None:
        reward += 5
    
    state = (dealer_hand, user_hand)

    return state, reward, done

# Q-LEARNING

def learn(ALPHA, GAMMA, q_value, max_next_q_value, reward):
    return (1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * max_next_q_value)

def init_q_table(seed=42):
    # np.random.seed(seed)

    # State space size
    player_sum_space_size = 32  # 0 to 21 inclusive
    dealer_upcard_space_size = 32  # 0 to 21 inclusive

    # Action space size
    action_space_size = 2  # 0 for stand, 1 for draw

    # q_table = np.random.uniform(low=-1, high=1, size=(player_sum_space_size, dealer_upcard_space_size, action_space_size))
    # Initialize Q-table with small random values
    q_table = np.random.rand(player_sum_space_size, dealer_upcard_space_size, action_space_size) * 0.01

    return q_table




# TEST

q_table = init_q_table()
q_1710 = [(q_table[17][10][0], q_table[17][10][1])]
episodes = 10000
epochs = 100

EPSILON = 0.7
EPSILON_START = EPSILON
EPSILON_END = 0.025
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / epochs)

ALPHA = 0.01
GAMMA = 0.95


results = []
for epoch in range(epochs):
    epoch_results = []
    deck = genDeck()

    state = ([draw_card(deck)], [draw_card(deck)])

    for i in range(episodes):
        old_state = state
        # action = np.argmax(q_table[sum_hand(state[0])][sum_hand(state[1])])
        q_values = q_table[sum_hand(state[0])][sum_hand(state[1])]
        if not q_values[0] == q_values[1]:
            action = np.argmax(q_table[sum_hand(state[0])][sum_hand(state[1])])
        else:
            action = np.random.randint(0, 1)

        EPSILON *= EPSILON_DECAY
        if np.random.uniform(0, 1) < EPSILON:
            action = random.randint(0, 1)

        state, reward, done = play(state, action, deck)

        q_table[sum_hand(state[0])][sum_hand(state[1])][action] = learn(ALPHA, GAMMA, q_table[sum_hand(old_state[0])][sum_hand(old_state[1])][action], np.max(q_table[sum_hand(state[0])][sum_hand(state[1])]), reward)
        if sum_hand(state[0]) == 17 and sum_hand(state[1]) == 10:
            q_1710.append((q_table[17][10][0], q_table[17][10][1]))
        if done:
            win = True if reward >= 1 else False
            epoch_results.append(win)
            state = ([draw_card(deck)], [draw_card(deck)])
            

        if i % 10 == 0:
            deck = genDeck()
    results.append(epoch_results)



#print(results)
wr = [sum(result) / len(result) for result in results]
#print(wr)
plt.plot(np.linspace(1, epochs, epochs), wr)
print("first:", wr[:3], "\nlast:", wr[-3::])


# Transpose the list of tuples to get separate lists for each value
q_values_a, q_values_b = zip(*q_1710)

# Plot both sets of values in the same plot
plt.plot(np.linspace(1, len(q_values_a), len(q_values_a)), q_values_a, label='Action 0 Q-Values')
plt.plot(np.linspace(1, len(q_values_b), len(q_values_b)), q_values_b, label='Action 1 Q-Values')

# plt.ylim(2, -300)
# plt.xlim(0, 200)

# Add labels and legend
plt.xlabel('Episode')
plt.ylabel('Q-Values')
plt.legend()

# Show the plot
plt.show()