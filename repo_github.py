import json
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from urllib.request import urlopen

def import_repo(search):
	with urlopen(f'https://api.github.com/search/repositories?q={search}') as resp:
		raw = resp.read()
	return json.loads(raw)['items']

def import_stuff():
	
	repos = [import_repo(f'language:python&sort=stars&order=desc&per_page=100&page={i}') for i in list(range(1,5))]
	new_repos = [i for repo in repos for i in repo]
	random.shuffle(new_repos)
	l_forks = [i['forks_count'] for i in new_repos]
	l_stars = [i['stargazers_count'] for i in new_repos]	
	forks = []
	stars = []
	for s in range(len(l_stars)):
		if l_stars[s]/l_forks[s] < 9:
			stars.append(l_stars[s])
			forks.append(l_forks[s])	
	return np.array(forks), np.array(stars) 
	

def model(forks,stars):
	forks_train = forks[:250] 
	forks_test = forks[250:]

	stars_train = stars[:250]
	stars_test = stars[250:]

	regr = linear_model.LinearRegression()
	regr.fit(stars_train.reshape(-1,1), forks_train.reshape(-1,1))

	forks_pred = regr.predict(stars_test.reshape(-1,1))

	plt.scatter(stars_test, forks_test,  color='black')
	plt.plot(stars_test, forks_pred, linewidth=2)
	plt.show()

	success = r2_score(forks_test, forks_pred)
	print(f"success of our model is: {success}")
	print('Coefficients: ', regr.coef_)

forks, stars = import_stuff()
model(forks,stars)
