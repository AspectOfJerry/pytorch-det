import timeit


# Define the first function
def function1():
    pass


# Define the second function
def function2():
    pass


# Measure time for function1
time_function1 = timeit.timeit('function1()', globals=globals(), number=1000)
time_function2 = timeit.timeit('function2()', globals=globals(), number=1000)

print(f"Execution time for function1: {time_function1:.6f} seconds")
print(f"Execution time for function2: {time_function2:.6f} seconds")

if time_function1 < time_function2:
    print("Function 1 is faster!")
elif time_function2 < time_function1:
    print("Function 2 is faster!")
else:
    print("Both functions are ridiculously precisely equally fast!")
