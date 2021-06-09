with open("coverage.xml", "r") as f:
    data = f.read()

data.replace("kernels/elementwise.h", "kernels/elementwise.h")