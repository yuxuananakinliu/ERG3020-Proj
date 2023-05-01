```python
## Imports
import pandas as pd
import numpy as np
import networkx as nx
import pickle
```


```python
## Read data
movies = pd.read_csv('work/movies.csv')
ratings = pd.read_csv('work/ratings.csv')
```


```python
movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9737</th>
      <td>193581</td>
      <td>Black Butler: Book of the Atlantic (2017)</td>
      <td>Action|Animation|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>9738</th>
      <td>193583</td>
      <td>No Game No Life: Zero (2017)</td>
      <td>Animation|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>9739</th>
      <td>193585</td>
      <td>Flint (2017)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>9740</th>
      <td>193587</td>
      <td>Bungo Stray Dogs: Dead Apple (2018)</td>
      <td>Action|Animation</td>
    </tr>
    <tr>
      <th>9741</th>
      <td>193609</td>
      <td>Andrew Dice Clay: Dice Rules (1991)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
<p>9742 rows × 3 columns</p>
</div>




```python
## Construct the rating map
user = ratings['userId'].unique()
r_map_nan = pd.DataFrame(columns=movies['movieId'].to_list(), index=user)
for i in range(len(ratings)):
    ind = ratings['userId'][i]
    col = ratings['movieId'][i]
    r_map_nan.at[ind, col] = ratings['rating'][i]

r_map = r_map_nan.fillna(0)
r_map.head().iloc[:, :15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = r_map.iloc[:, :1000]
test_nan = r_map_nan.iloc[:, :1000]
```


```python
## Similarity calculation function
def eud(base, x, y):
    x_r = base.loc[x].values
    y_r = base.loc[y].values
    eudist = 1 / (1 + np.linalg.norm(x_r - y_r))
    return eudist


## Construct similarity graph
def user_sim(r_matrix):
    W = pd.DataFrame(columns=user, index=user)
    for u in user:
        for v in user:
            if u == v:
                W.at[u, v] = 0
            else:
                W.at[u, v] = round(eud(r_matrix, u, v), 3)
    W.to_csv('work/user_similarity_matrix', sep=',')
    return W

#user_sim(test)
```


```python
## Weight normalization
def norm_W(w):
    for id in range(len(w)):
        summ = sum(w.iloc[id, :])
        for z in range(len(w)):
            ori = w.iloc[id, :].iat[z]
            w.iloc[id, :].iat[z] = ori/summ
    w.to_csv('work/user_similarity_matrix_normalized', sep=',')
    return w

#W = pd.read_csv('work/user_similarity_matrix', sep=',', index_col=0)
#norm_w = norm_W(W)
#norm_w.iloc[:, :10]
```


```python
## NaN detection function
def de_nan(base, rx, row):
    ori = base.iloc[rx].index.to_list()
    cle = row.index.to_list()
    for dex in cle:
        ori.remove(dex)
    return ori


## Reconstruction function
def recons(new_data, template):
    temp = template
    for u in range(len(new_data)):
        for i in range(len(new_data[u][0])):
            temp.at[u+1, new_data[u][0][i]] = new_data[u][1][i]
    return temp


## Nearest-neighbor collaborative filtering
def NNCF(rates, user_sim_m):
    new_map = rates  # Copy the unfinished rating map
    new_rates = []  # New rating collection

    for k in range(len(rates)):
        row = rates.iloc[k].dropna()
        simi = user_sim_m.iloc[k].sort_values(ascending=False).index.to_list()  # Sorted similar users
        if len(row) == 0:
            new_map.iloc[k, :] = 2.8 * np.ones(len(rates.iloc[k]))
            new_rates.append(rates.iloc[k].index.tolist())
        else:
            avg_x = sum(row) / len(row)
            unk = de_nan(rates, k, row)  # Unrated movies
            new_rates.append(unk)

            for va in unk:
                temp = 0
                count = 0
                nei = 0
                weight_sum = 0

                while count < 10 and nei < len(simi):
                    sco = rates.loc[int(simi[nei]), va]
                    if np.isnan(sco):
                        pass
                    else:
                        count += 1
                        weight_sum += user_sim_m.loc[k+1, simi[nei]]
                        neighbor = rates.loc[int(simi[nei])].dropna()
                        cur_avg = sum(neighbor) / len(neighbor)
                        temp += (sco - cur_avg) * user_sim_m.loc[k+1, simi[nei]]

                    nei += 1

                if weight_sum == 0:
                    point = round(avg_x, 1)
                else:
                    temp = temp / weight_sum
                    point = round(avg_x + temp, 1)

                new_map.loc[k+1, va] = point
            print('\r', k + 1, end='', flush=True)

    new_map.to_csv('work/new_map', sep=',')
    return new_map, new_rates


W = pd.read_csv('work/user_similarity_matrix', sep=',', index_col=0)
matrix, inds = NNCF(test_nan, W)
matrix.iloc[:, :15]
```

    /environment/miniconda3/lib/python3.7/site-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_block(indexer, value, name)
    /environment/miniconda3/lib/python3.7/site-packages/pandas/core/indexing.py:723: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      iloc._setitem_with_indexer(indexer, value, self.name)


     610




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>4.2</td>
      <td>4.0</td>
      <td>3.3</td>
      <td>3.9</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.7</td>
      <td>3.8</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.8</td>
      <td>4.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.9</td>
      <td>3.3</td>
      <td>3.4</td>
      <td>2.4</td>
      <td>3.4</td>
      <td>4.2</td>
      <td>2.7</td>
      <td>2.9</td>
      <td>3.0</td>
      <td>3.4</td>
      <td>3.8</td>
      <td>2.4</td>
      <td>3.1</td>
      <td>3.7</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.5</td>
      <td>1.1</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>1.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>-0.0</td>
      <td>0.8</td>
      <td>1.5</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.9</td>
      <td>3.4</td>
      <td>3.2</td>
      <td>2.7</td>
      <td>3.5</td>
      <td>4.3</td>
      <td>3.0</td>
      <td>3.1</td>
      <td>3.2</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>2.4</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>3.7</td>
      <td>3.3</td>
      <td>2.6</td>
      <td>3.5</td>
      <td>4.1</td>
      <td>3.2</td>
      <td>3.1</td>
      <td>3.0</td>
      <td>3.1</td>
      <td>4.1</td>
      <td>2.4</td>
      <td>3.3</td>
      <td>4.0</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>606</th>
      <td>2.5</td>
      <td>4.0</td>
      <td>3.7</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.4</td>
      <td>2.5</td>
      <td>3.5</td>
      <td>3.2</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>3.1</td>
      <td>3.7</td>
      <td>4.2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>607</th>
      <td>4.0</td>
      <td>4.1</td>
      <td>3.8</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>3.1</td>
      <td>3.5</td>
      <td>3.2</td>
      <td>3.3</td>
      <td>3.0</td>
      <td>3.1</td>
      <td>3.7</td>
      <td>4.1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>608</th>
      <td>2.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.9</td>
      <td>2.9</td>
      <td>3.4</td>
      <td>2.0</td>
      <td>2.4</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.2</td>
      <td>2.0</td>
      <td>2.6</td>
      <td>3.1</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>609</th>
      <td>3.0</td>
      <td>3.5</td>
      <td>3.2</td>
      <td>2.3</td>
      <td>3.4</td>
      <td>3.9</td>
      <td>2.5</td>
      <td>2.9</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>3.6</td>
      <td>2.4</td>
      <td>3.0</td>
      <td>3.6</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>610</th>
      <td>5.0</td>
      <td>3.9</td>
      <td>4.1</td>
      <td>3.4</td>
      <td>4.3</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.6</td>
      <td>4.3</td>
      <td>4.7</td>
      <td>3.6</td>
      <td>4.1</td>
      <td>4.6</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 15 columns</p>
</div>




```python
## Genre similarity function
def genre_simi(a, b):
        overlap = 0
        for gens in a:
            if gens in b:
                overlap += 1
        all_included = len(np.unique(np.array(a + b)))
        sim = overlap / all_included

        return sim
```


```python
## Personalized PageRank
def Personalized_PageRank(ori_table, target_ind):
    rec = []  # Recommendation order
    for us in range(len(target_ind)):
        orig_sco = []  # Original Pr for Page Rank
        sco_dict = {}
        relation = nx.Graph()  # Temperory movie similarity graph

        for t in range(len(target_ind[us])):
            mov_id = int(target_ind[us][t])

            orig_sco.append(ori_table.loc[us+1, str(mov_id)])

            gen = movies.loc[mov_id, 'genres']
            gen_type = gen.split('|')
            for f in range(t + 1, len(target_ind[us])):
                genn = movies.loc[int(target_ind[us][f]), 'genres']
                genn_type = genn.split('|')
                sim = genre_simi(gen_type, genn_type)
                if sim != 0.0:
                    relation.add_edge(movies.loc[mov_id, 'movieId'], 
                                    movies.loc[int(target_ind[us][f]), 'movieId'], 
                                    weight=sim
                                    )
                    
        orig_sco = np.array(orig_sco)
        orig_sco = orig_sco / orig_sco.sum()
        for o in range(len(orig_sco)):
            sco_dict[int(target_ind[us][o])] = orig_sco[o]

        r = nx.pagerank(relation, alpha=0.85, personalization=sco_dict)  # Personalized PageRank
        sorted_r = sorted(r.items(), key=lambda y: y[1], reverse=True)
        rec.append(dict(sorted_r))

    return rec


## PageRank data preprocessing
filled_map = pd.read_csv('work/new_map', sep=',', index_col=0)

rec_rst = Personalized_PageRank(filled_map, inds)
```


```python
rec_table = pd.DataFrame(index=user, columns=[top for top in range(1, 101)])
for i in range(len(user)):
    rec_table.iloc[i] = list(rec_rst[i].keys())[:100]

rec_table  # The top 100 recommended movies' id for Users
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1243</td>
      <td>218</td>
      <td>272</td>
      <td>1292</td>
      <td>82</td>
      <td>106</td>
      <td>324</td>
      <td>562</td>
      <td>568</td>
      <td>994</td>
      <td>...</td>
      <td>1132</td>
      <td>1231</td>
      <td>261</td>
      <td>299</td>
      <td>321</td>
      <td>835</td>
      <td>944</td>
      <td>1051</td>
      <td>1103</td>
      <td>1228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1292</td>
      <td>1243</td>
      <td>568</td>
      <td>984</td>
      <td>1300</td>
      <td>82</td>
      <td>106</td>
      <td>324</td>
      <td>562</td>
      <td>994</td>
      <td>...</td>
      <td>121</td>
      <td>299</td>
      <td>306</td>
      <td>321</td>
      <td>835</td>
      <td>1051</td>
      <td>1225</td>
      <td>476</td>
      <td>14</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1236</td>
      <td>496</td>
      <td>495</td>
      <td>1292</td>
      <td>838</td>
      <td>148</td>
      <td>1243</td>
      <td>58</td>
      <td>82</td>
      <td>106</td>
      <td>...</td>
      <td>337</td>
      <td>385</td>
      <td>536</td>
      <td>679</td>
      <td>944</td>
      <td>1132</td>
      <td>956</td>
      <td>1046</td>
      <td>608</td>
      <td>472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1243</td>
      <td>1292</td>
      <td>82</td>
      <td>272</td>
      <td>324</td>
      <td>568</td>
      <td>984</td>
      <td>994</td>
      <td>1300</td>
      <td>96</td>
      <td>...</td>
      <td>341</td>
      <td>448</td>
      <td>491</td>
      <td>926</td>
      <td>1231</td>
      <td>26</td>
      <td>326</td>
      <td>354</td>
      <td>524</td>
      <td>685</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1292</td>
      <td>82</td>
      <td>106</td>
      <td>324</td>
      <td>568</td>
      <td>1243</td>
      <td>984</td>
      <td>1300</td>
      <td>1236</td>
      <td>218</td>
      <td>...</td>
      <td>385</td>
      <td>679</td>
      <td>944</td>
      <td>1051</td>
      <td>1104</td>
      <td>1228</td>
      <td>211</td>
      <td>341</td>
      <td>536</td>
      <td>926</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>606</th>
      <td>568</td>
      <td>106</td>
      <td>516</td>
      <td>562</td>
      <td>96</td>
      <td>218</td>
      <td>322</td>
      <td>371</td>
      <td>430</td>
      <td>635</td>
      <td>...</td>
      <td>537</td>
      <td>722</td>
      <td>608</td>
      <td>43</td>
      <td>219</td>
      <td>254</td>
      <td>510</td>
      <td>523</td>
      <td>636</td>
      <td>162</td>
    </tr>
    <tr>
      <th>607</th>
      <td>1292</td>
      <td>82</td>
      <td>106</td>
      <td>324</td>
      <td>568</td>
      <td>984</td>
      <td>1243</td>
      <td>1300</td>
      <td>516</td>
      <td>583</td>
      <td>...</td>
      <td>1225</td>
      <td>778</td>
      <td>299</td>
      <td>306</td>
      <td>385</td>
      <td>536</td>
      <td>679</td>
      <td>926</td>
      <td>959</td>
      <td>1104</td>
    </tr>
    <tr>
      <th>608</th>
      <td>99</td>
      <td>1292</td>
      <td>246</td>
      <td>363</td>
      <td>77</td>
      <td>108</td>
      <td>602</td>
      <td>1189</td>
      <td>162</td>
      <td>116</td>
      <td>...</td>
      <td>306</td>
      <td>385</td>
      <td>536</td>
      <td>679</td>
      <td>926</td>
      <td>1104</td>
      <td>1132</td>
      <td>1228</td>
      <td>319</td>
      <td>14</td>
    </tr>
    <tr>
      <th>609</th>
      <td>1292</td>
      <td>82</td>
      <td>106</td>
      <td>324</td>
      <td>568</td>
      <td>984</td>
      <td>1243</td>
      <td>1300</td>
      <td>516</td>
      <td>583</td>
      <td>...</td>
      <td>62</td>
      <td>299</td>
      <td>306</td>
      <td>385</td>
      <td>536</td>
      <td>679</td>
      <td>926</td>
      <td>944</td>
      <td>959</td>
      <td>1104</td>
    </tr>
    <tr>
      <th>610</th>
      <td>1292</td>
      <td>106</td>
      <td>324</td>
      <td>568</td>
      <td>1243</td>
      <td>82</td>
      <td>516</td>
      <td>583</td>
      <td>973</td>
      <td>994</td>
      <td>...</td>
      <td>1104</td>
      <td>1193</td>
      <td>1225</td>
      <td>211</td>
      <td>213</td>
      <td>306</td>
      <td>341</td>
      <td>536</td>
      <td>959</td>
      <td>1132</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 100 columns</p>
</div>




```python
rec_table.to_csv('work/top-100 recommended', sep=',')
```


```python
def list2txt(li):
    with open(r'work/pagerank_result.txt', 'w') as fp:
        for item in li:
            # write each item on a new line
            fp.write("%s\n" % str(item))
        print('Done')

list2txt(rec_rst)
```

    Done



```python

```
