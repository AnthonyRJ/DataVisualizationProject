import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#riotwatcher is usefull to extract data
from riotwatcher import LolWatcher, ApiError #pip install riotwatcher
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

############################### GLOBAL VARIABLES FOR THE API ###############################

api_key = 'RGAPI-6a03e779-8f7a-41dd-b2b5-64c0782cbc8f' #change every 24h
watcher = LolWatcher(api_key)
region = 'euw1'

################ COLLECTING, CLEANING & PREPARE DATA FROM THE FIRST DATASET ################

#First dataset for general data on the game
df = pd.read_csv("./data/average_50k_games.csv")

#deleting columns that i think it's not usefull
df = df.drop(columns = ["gameId", "seasonId"])

#most recent version of each data from the API
versions = watcher.data_dragon.versions_for_region(region)

#playable character informations of the game
rawChampions = watcher.data_dragon.champions(versions["n"]["champion"])
cleanChampions = {}
for champion in rawChampions["data"]:
    cleanChampions[int(rawChampions["data"][champion]["key"])] = [rawChampions["data"][champion]["name"], rawChampions["data"][champion]["tags"][0]]

#playable summoner spells informations of the game
rawSummonerSpells = watcher.data_dragon.summoner_spells(versions["n"]["summoner"])
cleanSummonerSpells = {}
for sumSpell in rawSummonerSpells["data"]:
    cleanSummonerSpells[int(rawSummonerSpells["data"][sumSpell]["key"])] =  rawSummonerSpells["data"][sumSpell]["name"]
    

#the champions that each player picked for the game
pickedChampions = pd.DataFrame(data = df[["t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id","t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id","t2_champ5id"]])

#the champions that each player banned for the game
bannedChampions = pd.DataFrame(data = df[['t1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5','t2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5']])

#the 2 summoner spells that each player picked for the game
pickedSummonerSpells = pd.DataFrame(data = df[['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2','t1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1',
                        't1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2','t2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                        't2_champ5_sum1','t2_champ5_sum2']])

#Replacing ids by their name
def idsToChampionName(id, champions):
    # -1 means that the player banned nothing (don't ban a champion is a choice)
    if(id == -1):
        return "NoBan"
    return champions[id][0]

def idsToSummonerSpellsName(id, spells):
    return spells[id]

for column in pickedChampions:
    pickedChampions[column] = pickedChampions[column].apply(lambda x:  idsToChampionName(x,cleanChampions))  

for column in bannedChampions:
    bannedChampions[column] = bannedChampions[column].apply(lambda x:  idsToChampionName(x,cleanChampions))

for column in pickedSummonerSpells:
    pickedSummonerSpells[column] = pickedSummonerSpells[column].apply(lambda x : idsToSummonerSpellsName(x, cleanSummonerSpells))


#Champion picks and bans
picks = pd.concat([pickedChampions['t1_champ1id'],pickedChampions['t1_champ2id'],pickedChampions['t1_champ3id'],pickedChampions['t1_champ4id'],pickedChampions['t1_champ5id'],
                      pickedChampions['t2_champ1id'],pickedChampions['t2_champ2id'],pickedChampions['t2_champ3id'],pickedChampions['t2_champ4id'],pickedChampions['t2_champ5id']])
sortedPicks = sorted(picks)


bans = pd.concat([bannedChampions['t1_ban1'],bannedChampions['t1_ban2'],bannedChampions['t1_ban3'],bannedChampions['t1_ban4'],bannedChampions['t1_ban5'],
                     bannedChampions['t2_ban1'],bannedChampions['t2_ban2'],bannedChampions['t2_ban3'],bannedChampions['t2_ban4'],bannedChampions['t2_ban5']])
sortedBans = sorted(bans)

#showing most played role and most used summoner spells
sumSpellsColumns = ['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2','t1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2',
                    't2_champ1_sum1','t2_champ1_sum2','t2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2','t2_champ5_sum1','t2_champ5_sum2']

spells = pickedSummonerSpells[sumSpellsColumns].apply(pd.value_counts)
spells["count"] = spells[sumSpellsColumns].sum(axis=1)

#First objectives per team data for plot

#Blue and Red are the team color
blueRedData = df.replace([0,1,2],['neither','blue','red'])

labels = ['firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
fTotals = blueRedData[labels].apply(pd.value_counts)

indexes = ['blue','red','neither']
sortedTotals = fTotals.reindex(index=indexes)

#Colors corresponding to blue and red team. For "neither" it's grey
teamColors = ['#3498DB','#E74C3C','#BDC3C7']

################ COLLECTING, CLEANING & PREPARE DATA FROM THE SECOND DATASET ################
df2 = pd.read_csv("./data/high_ranked_games.csv")

############################### STREAMLIT PAGE ###############################

st.sidebar.title("Menu")
select = st.sidebar.radio("",["Pre-game analysis", "In-game analysis"])

if select == "Pre-game analysis":
    st.markdown("<h3>REINO JOAQUIM <br> Anthony <br> DS3</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Analyzing League of Legends data</h1>", unsafe_allow_html=True)

    #League of legends presentation
    st.markdown("<h3>What is league of legends ?</h3>", unsafe_allow_html=True)
    st.write("Before watching my visualizations, i would recommend you to watch this short video to understand how the game works :")
    st.markdown("<a> https://www.youtube.com/watch?v=BGtROJeMPeE </a>", unsafe_allow_html=True)
    #Showing the data I used for the next visualizations
    st.markdown("<h3>Data overview</h3>", unsafe_allow_html=True)
    st.write(df.head())

    #showing picks versus bans with a countplot 
    st.markdown("<h3>Picked and banned champions</h3><br>", unsafe_allow_html=True)
    figPickBan, (a1, a2) = plt.subplots(ncols=2, figsize=(18,35))
    a1.set_title("Picked champion")
    a2.set_title("Banned champion")
    sns.countplot(y=sortedPicks, data = pickedChampions, ax = a1)
    sns.countplot(y=sortedBans, data = bannedChampions, ax = a2)
    plt.xticks(rotation = 90)
    st.pyplot(figPickBan.figure)

    #showing most used summoner spells with a pie chart 
    st.markdown("<h3>Most used summoner spells</h3>", unsafe_allow_html=True)
    fig = px.bar(spells["count"], x = spells.index, y = spells["count"], color = spells["count"])
    st.plotly_chart(fig)

    #showing objectives per team
    st.markdown("<h3>First objectives per side</h3><br>", unsafe_allow_html=True)

    rows, cols = 2,3
    fig = plt.figure(figsize=(15,10))

    #Ploting the 6 pie charts
    i = 1
    while(i < 7):
        ax = fig.add_subplot(rows,cols,i)
        plt.pie(sortedTotals[sortedTotals.columns[i-1]], colors = teamColors, radius = 1.1, autopct="%.2f")
        ax.set_title(labels[i-1])
        i += 1
    st.pyplot(fig.figure)

elif select == "In-game analysis":
    st.markdown("<h3>REINO JOAQUIM <br> Anthony <br> DS3</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Analyzing League of Legends data</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Data overview</h3>", unsafe_allow_html=True)
    st.write(df2.head())

    st.markdown("<h3>Red team features correlations heatmap</h3><br>", unsafe_allow_html=True)

    #showing correlations between features for the red team (we could do the same for blue team but it would be the same values)
    corr = df2[[column for column in df2.columns if 'blue' not in column]].corr()

    fig,ax = plt.subplots(figsize=(18, 18))
    heatmap = sns.heatmap(corr,annot=True,fmt=".1f",cbar=False, annot_kws={'size':18}, cmap='coolwarm')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize = 18)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize = 18)
    st.pyplot(heatmap.figure)
    
    #Showing correlations victory
    st.markdown("<h3>Win / Lose correlations</h3><br>", unsafe_allow_html=True)

    #blue team victory correlations
    blue_win_condition = df2[[col for col in df2.columns if col != 'blueWins']].corrwith(df2['blueWins']).to_frame().sort_values(ascending = False, by = 0)
    blue_win_condition = pd.concat([blue_win_condition.head(7), blue_win_condition.tail(7)])
    blue_win_condition.columns = ['Blue Win Corr']

    #red team victory correlations
    red_win_condition = df2[[col for col in df2.columns if col != 'blueWins']].corrwith(df2['blueWins'].map({0:1, 1:0})).to_frame().sort_values(ascending = False, by = 0)
    red_win_condition = pd.concat([red_win_condition.head(7), red_win_condition.tail(7)])
    red_win_condition.columns = ['Red Win Corr']

    fig = plt.figure(figsize=(20,15)) 
    axe1 = fig.add_subplot(121)
    axe2 = fig.add_subplot(122)

    plt.figure(figsize=(8,8))
    ht1 = sns.heatmap(blue_win_condition,cmap='coolwarm', annot=True, annot_kws={'size':17}, ax = axe1)
    ht2 = sns.heatmap(red_win_condition,cmap='coolwarm', annot=True, annot_kws={'size':17}, ax = axe2)
    ht1.set_xticklabels(ht1.get_xticklabels(), fontsize = 12)
    ht1.set_yticklabels(ht1.get_yticklabels(), fontsize = 12)
    ht2.set_xticklabels(ht2.get_xticklabels(), fontsize = 12)
    ht2.set_yticklabels(ht2.get_yticklabels(), fontsize = 12)
    st.pyplot(fig.figure)

    #The violin plot for golds
    mask = ['blueTotalGold', 'redTotalGold']
    blueSide = df2[mask[0:1]]
    blueSide.columns = ['TotalGold']
    
    redSide = df2[mask[1:2]]
    redSide.columns = ['TotalGold']

    blueSide['TeamSide'] = 'Blue team'
    redSide['TeamSide'] = 'Red team'
    
    totalGolds = pd.concat([blueSide, redSide])
    fig = px.box(totalGolds, y = "TotalGold", color = "TeamSide", title = 'Total gold per side')
    st.plotly_chart(fig)
