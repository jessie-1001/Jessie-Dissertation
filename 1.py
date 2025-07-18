import sys
import arch
from arch import arch_model
import numpy as np

print("Python version:", sys.version)
print("arch version:", arch.__version__)
print("arch package location:", arch.__file__)

# Test .dist availability
returns = np.random.normal(0, 1, size=300)
model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
res = model.fit(disp="off")

print("Does model result have .dist attribute?", hasattr(res, "dist"))
