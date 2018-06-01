# Bandits

## Stationary Environment ##

### Comparison from the book ###

![plot_from_book_1](images/book_1_rewards.svg)
![plot_from_book_2](images/book_1_actions.svg)

```
python -m scripts.compare_bandits_stationary images/book_1 -a epsilon epsilon epsilon -s 0.0 0.01 0.1 -l "ε=0", "ε=0.01" "ε=0.1" -t "ε-greedy bandits"
```

### Epsilon-greedy bandits ###

![epsilon_1](images/epsilon_rewards.svg)
![epsilon_2](images/epsilon_actions.svg)

```
python -m scripts.compare_bandits_stationary images/epsilon -a epsilon epsilon epsilon epsilon epsilon epsilon -s 0.0 0.01 0.1 0.2 0.5 1.0 -l "ε=0" "ε=0.01" "ε=0.1" "ε=0.2" "ε=0.5" "ε=1.0" -t "ε-greedy bandits"
```

### Softmax bandits ###

![softmax_1](images/softmax_rewards.svg)
![softmax_2](images/softmax_actions.svg)

```
python -m scripts.compare_bandits_stationary images/softmax -a softmax softmax softmax softmax softmax -s 0.1 0.2 0.5 1.0 2.0 -l "τ=0.1" "τ=0.2" "τ=0.5" "τ=1.0" "τ=2.0" -t "softmax bandits"
```

### Optimistic initialization ###

![optimistic_init_1](images/optimistic_init_rewards.svg)
![optimistic_init_2](images/optimistic_init_actions.svg)

```
python -m scripts.compare_bandits_stationary images/optimistic_init -a epsilon epsilon -s 0.0 0.1 -i 5.0 0.0 -l "ε=0, init=5" "ε=0.1, init=0" -t "Optimistic Initialization"
```

### Final Comparison ###

![epsilon_vs_softmax_vs_optimistic_1](images/epsilon_vs_softmax_vs_optimistic_rewards.svg)
![epsilon_vs_softmax_vs_optimistic_2](images/epsilon_vs_softmax_vs_optimistic_actions.svg)

```
python -m scripts.compare_bandits_stationary images/epsilon_vs_softmax_vs_optimistic -a epsilon epsilon softmax -s 0.1 0.0 0.2 -l "ε=0.1, init=0", "ε=0, init=5" "τ=0.2, init=0" -i 0.0 5.0 0.0
```

## Non-stationary Environment ##

![non_stationary_bandits_1](images/nonstationary_rewards.svg)
![non_stationary_bandits_2](images/nonstationary_actions.svg)

```
python -m scripts.compare_bandits_nonstationary images/nonstationary -a epsilon epsilon -s 0.1 0.1 --alphas 0.0 0.1 -l "α=1/k" "α=0.1" -t "ε-greedy bandits, ε=0.1"
```