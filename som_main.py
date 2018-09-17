import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import somnetwork as SOM
from collections import Counter

def votes():
	
	layer_dim = [31, 100]
	som = SOM.SOMNetwork(layer_dim, mp=True)
	som.train(mp_votes, 100)
	result = som.predict(mp_votes)
	

votes()