from sort_and_analyze_beatmaps import is_crossover, is_spin, is_double_step, is_footswitch

# L = 0, D = 1, U = 2, R = 3
# Crossover is a sideways candle, L D R L U R
# L(0) -> D(1) -> R(3)
test1 = [[0], [1], [3]] 
print(f"L D R: crossover={is_crossover(test1)}")

test2 = [[3], [1], [0]] 
print(f"R D L: crossover={is_crossover(test2)}")

test3 = [[2], [0], [1]] 
print(f"U L D: crossover={is_crossover(test3)}")

test4 = [[0], [1], [3], [1], [0]]
print(f"L D R D L: crossover={is_crossover(test4)}")
