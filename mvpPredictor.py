# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:49:35 2021

@author: conno
"""
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from operator import itemgetter
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from itertools import chain


def scrape_mvp_data(daterange):
    
    mvpdata = pd.DataFrame()

    for y in daterange:
        year = y
        
        
        url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
        
        html = urlopen(url)
        
        soup = BeautifulSoup(html, features='lxml')
        
        table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="mvp")
        
        headers = [th.getText() for th in table.findAll('tr', limit=2)[1].findAll('th')]
        headers = headers.remove('Rank')
        rows = table.findAll('tr')[2:]
        rows_data = [[td.getText() for td in rows[i].findAll('td')]
                            for i in range(len(rows))]
        
        yearlydata = pd.DataFrame(rows_data, columns = headers)
        
        yearlydata['Year'] = year
        
        mvpdata = mvpdata.append(yearlydata)
        
    mvpdata.columns = ['Player', 'Age', 'Tm', 'First', 'Pts Won', 'Pts Max', 'Share',
                       'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48', 'Year']
    return mvpdata

def scrape_vorp_data(daterange):
    
    advdata = pd.DataFrame()

    
    for y in daterange:
        year = y
        
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        
        html = urlopen(url)
        
        soup = BeautifulSoup(html, features='lxml')
        
        table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="advanced_stats")
        
        headers = [th.getText() for th in table.findAll('tr', limit=2)[0].findAll('th')]
        headers = headers.remove('Rk')
        rows = table.findAll('tr')[1:]
        rows_data = [[td.getText() for td in rows[i].findAll('td')]
                            for i in range(len(rows))]
        
        advyearly = pd.DataFrame(rows_data, columns = headers)
        
        advyearly['Year'] = year
        
        advdata = advdata.append(advyearly)
        
    
    return advdata

def scrape_reg_data(daterange):
    
    regdata = pd.DataFrame()

    
    for y in daterange:
        year = y
        
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        
        html = urlopen(url)
        
        soup = BeautifulSoup(html, features='lxml')
        
        table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="per_game_stats")
        
        headers = [th.getText() for th in table.findAll('tr', limit=1)[0].findAll(['th', 'tr'])]
        rows = table.findAll('tr')[1:]
        rows_data = [[td.getText() for td in rows[i].findAll(['td', 'th'])]
                            for i in range(len(rows))]
        regyearly = pd.DataFrame(rows_data, columns = headers)
        
        
        regyearly['Year'] = year
        
        regdata = regdata.append(regyearly)
        
    
    return regdata

def scrape_seed_data(daterange):
    
    seeddata = pd.DataFrame()
    
    for y in daterange:
        year = y
        
        url = f'https://www.basketball-reference.com/friv/standings.fcgi?month=6&day=30&year={year}&lg_id=NBA'
        
        html = urlopen(url)
        
        soup = BeautifulSoup(html, features='lxml')
        
        table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="standings_e")
        
        headers = [th.getText() for th in table.findAll('tr', limit=2)[0].findAll('th')]
        #headers = headers.remove('Rk')
        rows = table.findAll(['tr', 'th'])[1:]
        rows_data = [[td.getText() for td in rows[i].findAll(['td', 'th'])]
                            for i in range(len(rows))]
        
        seedyearly = pd.DataFrame(rows_data, columns = headers)
        
        seedyearly = seedyearly.dropna()
        
        seedyearly['Year'] = year
        
        seeddata = seeddata.append(seedyearly)
        
        print('Iterate')
        
        table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="standings_w")
        
        headers = [th.getText() for th in table.findAll('tr', limit=2)[0].findAll('th')]
        #headers = headers.remove('Rk')
        rows = table.findAll(['tr', 'th'])[1:]
        rows_data = [[td.getText() for td in rows[i].findAll(['td', 'th'])]
                            for i in range(len(rows))]
        
        seedyearly = pd.DataFrame(rows_data, columns = headers)
        
        seedyearly = seedyearly.dropna()
        
        seedyearly['Year'] = year
        
        seeddata = seeddata.append(seedyearly)
        
    return seeddata
      
currentyear = date.today().year
threepointyear = 1979
years = []        
for i in range(currentyear - threepointyear ):
    years.append(1980+i)
    
team_dict = {'Atlanta Hawks'	:'ATL', 'Brooklyn Nets'	: 'BKN', 'New Jersey Nets' : 'NJN', 'Boston Celtics' :'BOS', 'New Orleans Hornets': 'NOH',
             'Charlotte Hornets' :'CHH', 'Charlotte Bobcats': 'CHB', 'Chicago Bulls' :'CHI', 'Cleveland Cavaliers' :'CLE', 
             'Dallas Mavericks'	:'DAL', 'Denver Nuggets' :'DEN', 'Detroit Pistons' :'DET',
             'Golden State Warriors' :'GSW', 'Houston Rockets' :'HOU', 'Indiana Pacers' :'IND', 
             'Los Angeles Clippers'	:'LAC', 'Los Angeles Lakers' :'LAL', 'Memphis Grizzlies' :'MEM', 
             'Miami Heat' :'MIA', 'Milwaukee Bucks' :'MIL', 'Minnesota Timberwolves'	:'MIN',
             'New Orleans Pelicans'	:'NOP', 'New York Knicks' :'NYK', 'Oklahoma City Thunder' :'OKC', 
             'Orlando Magic':'ORL','Philadelphia 76ers' :'PHI', 'Phoenix Suns' :'PHO',
             'Portland Trail Blazers' :'POR', 'Sacramento Kings':'SAC', 'Kansas City Kings': 'KCK', 'San Antonio Spurs' :'SAS',
             'Toronto Raptors':'TOR', 'Utah Jazz' :'UTA','Washington Wizards': 'WAS', 'Washington Bullets': 'WSB', 
             'Seattle SuperSonics': 'SEA', 'Vancouver Grizzlies': 'VAN', 'San Diego Clippers': 'SDC', 'New Orleans/Oklahoma City Hornets': 'NOK'
}


mvpdata = scrape_mvp_data(years)
advdata = scrape_vorp_data(years)
seeddata = scrape_seed_data(years)
regdata = scrape_reg_data(years)
seeddata = seeddata.fillna('')
advdata = advdata.drop([18, 23], axis = 1)
advdata.columns = ['Player', 'Pos',	'Age', 'Tm', 'G',	'MP',	'PER',	'TS%',	'3PAr',	'FTr',	'ORB%',	'DRB%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'TOV%',	'USG%', 'OWS',	'DWS',	'WS',	'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP', 'Year']
advdata.to_excel('C:/Users/conno/Documents/CIS450/orangedata.xlsx')
mvpdata['Player']=mvpdata['Player'].astype(str)
advdata['Player']=advdata['Player'].astype(str)
advdata['Player'] = advdata['Player'].str.replace('*','')
seeddata['Eastern Conference'] = seeddata['Eastern Conference'].str.replace('*','')
seeddata['Western Conference'] = seeddata['Western Conference'].str.replace('*','')
regdata['Player'] = regdata['Player'].str.replace('*','')
cols = ['Eastern Conference', 'Western Conference']
seeddata['Tm'] = seeddata[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
seeddata = seeddata[['Tm', 'W/L%', 'Year']]
seeddata['Tm'] = seeddata['Tm'].map(team_dict).fillna(seeddata['Tm'])
seeddata['Year'] = seeddata['Year'].apply(pd.to_numeric)
seeddata['W/L%'] = seeddata['W/L%'].apply(pd.to_numeric)
seeddata['Seed'] = seeddata.groupby('Year')['W/L%'].rank(ascending=False, method='max')



df = mvpdata.merge(advdata[['Player','Year','VORP','BPM']], on=['Player','Year'], how='left')
df = df.merge(seeddata[['Tm','Seed', 'Year']],on=['Tm', 'Year'], how ='left')
df = df.dropna(axis=0)

y = df['Share']
x = df[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', 'WS', 'VORP', 'Seed']]
x = x.apply(pd.to_numeric)
x = x.fillna(0)
y = y.apply(pd.to_numeric)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(xtrain)
X_testscaled=sc_X.transform(xtest)

rf = RandomForestRegressor(n_estimators=100)

rf.fit(xtrain, ytrain)

sorted_idx = rf.feature_importances_.argsort()
plt.barh(x.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

plt.figure(figsize=(16,8))
correlationmatrix = sns.heatmap(x.corr(), vmin=-1, vmax=1, annot=True)

correlationmatrix.set_title('Correlation Matrix', pad=12)


def scores(y, model):
    
    if model == 'knn':
        model.fit(xtrain, ytrain)
    else:
        model.fit(xtrain, ytrain.values.ravel())
    y_pred = model.predict(xtest)
    
    
    print("Mean squared error: %.3f" % mse(ytest, y_pred))
    print('R2 score: %.3f' % r2_score(ytest, y_pred))

    if model == 'knn':
        cvScore = cross_val_score(model, xtest, ytest, cv = 3, scoring = 'r2')
    else:
        cvScore = cross_val_score(model, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
    print("R2 cross validation score: %0.2f (+/- %0.2f)" % (cvScore.mean(), cvScore.std() * 2))
    
    for i in y_pred:
        y.append(i)
        
dtrain = xgb.DMatrix(xtrain, label=ytrain)

def bo_tune_xgb(max_depth, gamma, learning_rate):
    params = {'max_depth': int(max_depth),
              'gamma': gamma,
              'learning_rate':learning_rate,
              'subsample': 0.8,
              'eta': 0.1,
              'eval_metric': 'rmse'}
    cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5)
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 10),
                                             'gamma': (0, 1),
                                             'learning_rate':(0,1)
                                            })

xgb_bo.maximize(n_iter=5, init_points=8)

params = xgb_bo.max['params']


params['max_depth']= int(params['max_depth'])


xgbr = xgb.XGBRegressor(**params)

y_xgbr = []

scores(y_xgbr, xgbr)
        
dnn = MLPRegressor(
    solver='lbfgs',
    hidden_layer_sizes=100,
    max_iter=10000,
    activation='identity',
    learning_rate ='invscaling')

y_dnn = []

scores(y_dnn, dnn)

rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse')

y_rf = []

scores(y_rf, rf)

knn = neighbors.KNeighborsRegressor(n_neighbors = 9, weights = 'uniform')

y_knn = []

scores(y_knn, knn)




####################################################################################################

testframe = regdata[regdata['Year']==1980]
testframe = testframe.merge(advdata[['VORP', 'Player', 'Year', 'WS']], on=['Player', 'Year'], how='left')
testframe = testframe.merge(seeddata[['Tm','Seed', 'Year']],on=['Tm', 'Year'], how ='left')


testframe = testframe.dropna()

testframe['MP'] = testframe['MP'].apply(pd.to_numeric)

testframe = testframe[testframe.MP > 30]

testframe = testframe.drop_duplicates(subset=['Player'])

predict = testframe[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', 'WS', 'VORP', 'Seed']]
predictnames = testframe.iloc[:,1]

predict = predict.apply(pd.to_numeric)

dnnPredict = dnn.predict(predict)
dnnPredict = dnnPredict.tolist()


dnnListUnsorted = [[i, j] for i, j in zip(predictnames, dnnPredict)]
dnnDataUnsorted = [row[1] for row in dnnListUnsorted]
dnnList = sorted(dnnListUnsorted, key = itemgetter(1), reverse = True)

dnnData = [row[1] for row in dnnList]
dnnNames = [row[0] for row in dnnList]

x_dnn = np.arange(len(dnnData))

x_dnn = x_dnn.argsort()

print(dnnList[0:20])

rfPredict = rf.predict(predict)
rfPredict = rfPredict.tolist()



rfListUnsorted = [[i, j] for i, j in zip(predictnames, rfPredict)]
rfDataUnsorted = [row[1] for row in rfListUnsorted]
rfList = sorted(rfListUnsorted, key = itemgetter(1), reverse = True)

rfData = [row[1] for row in rfList]
rfNames = [row[0] for row in rfList]

x_rf = np.arange(len(rfData))

print(rfList[0:20])

xgbPredict = xgbr.predict(predict)
xgbPredict = xgbPredict.tolist()



xgbListUnsorted = [[i, j] for i, j in zip(predictnames, xgbPredict)]
xgbDataUnsorted = [row[1] for row in xgbListUnsorted]
xgbList = sorted(xgbListUnsorted, key = itemgetter(1), reverse = True)

xgbData = [row[1] for row in rfList]
xgbNames = [row[0] for row in rfList]

x_xgb = np.arange(len(xgbData))

print(xgbList[0:20])

knnPredict = knn.predict(predict)
knnPredict = knnPredict.tolist()



knnListUnsorted = [[i, j] for i, j in zip(predictnames, knnPredict)]
knnDataUnsorted = [row[1] for row in knnListUnsorted]
knnList = sorted(knnListUnsorted, key = itemgetter(1), reverse = True)

knnData = [row[1] for row in rfList]
knnNames = [row[0] for row in rfList]

x_knn = np.arange(len(knnData))

print(knnList[0:20])

results_dict = {}

def test_all_years(daterange):
    for y in daterange:
        testframe = regdata[regdata['Year']==y]
        testframe = testframe.merge(advdata[['VORP', 'Player', 'Year', 'WS']], on=['Player', 'Year'], how='left')
        testframe = testframe.merge(seeddata[['Tm','Seed', 'Year']],on=['Tm', 'Year'], how ='left')
        
        
        testframe = testframe.dropna()
        
        testframe['MP'] = testframe['MP'].apply(pd.to_numeric)
        
        testframe = testframe[testframe.MP > 30]
        
        testframe = testframe.drop_duplicates(subset=['Player'])
        
        predict = testframe[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', 'WS', 'VORP', 'Seed']]
        predictnames = testframe.iloc[:,1]
        
        predict = predict.apply(pd.to_numeric)
        
        
        knnPredict = knn.predict(predict)
        knnPredict = knnPredict.tolist()
        
        
        
        knnListUnsorted = [[i, j] for i, j in zip(predictnames, knnPredict)]
        knnDataUnsorted = [row[1] for row in knnListUnsorted]
        knnList = sorted(knnListUnsorted, key = itemgetter(1), reverse = True)
        
        knnData = [row[1] for row in rfList]
        knnNames = [row[0] for row in rfList]
        
        x_knn = np.arange(len(knnData))
        
        results_dict[y]=knnList[0:10]
        
test_all_years(years)

results_df = pd.DataFrame.from_dict(results_dict)