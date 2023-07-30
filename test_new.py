import multiprocessing as mp
import random
import time

def some_calculations(x: float) -> float:
    #time.sleep(.5)
    # some calculations are done
    return x**2

if __name__ == "__main__":
    x_list = [random.random() for _ in range(20)]
    start_time = time.perf_counter()
    with mp.Pool(2) as p:
        x_recalculated = p.map(some_calculations, x_list)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")