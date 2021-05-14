import json
import os
import re
import random
import sys, getopt

import pandas as pd
import numpy as np

questions_final = []
answers_final = []
context_final = []
team_stats = pd.read_csv("data/basketball_teams.csv")
hof_stats = pd.read_csv("data/basketball_hof.csv")
draft_stats = pd.read_csv("data/basketball_draft.csv")
coach_stats = pd.read_csv("data/basketball_awards_coaches.csv")
player_stats = pd.read_csv("data/player_data.csv")
season_stats = pd.read_csv("data/Seasons_Stats.csv")
playoff_stats = pd.read_csv("data/basketball_series_post.csv")
playerAwards_stats = pd.read_csv("data/basketball_awards_players.csv")
champion_stats = playoff_stats.loc[playoff_stats['round'] == "F"]
answer_templates = [
    "[player] scored [points] points in the [year] season.",
    "[player] is on [team].",
    "[player] received the [award] in [year].",
    "[coach] received the [award] in [year].",
    "[team] were the champions in [year].",
    "[winTeam] beat [loseTeam] in the [game] in [year].",
    "[player] was inducted into the Hall of Fame in [year].",
    "[player] was drafted number [draftNumber] in [year].",
    "[player] played for [school] before the NBA.",
    "[team] had a record of [wins] wins and [loses] loses in the [time] season."
]
with open("data/questions.txt") as f:
    question_templates = f.readlines()
question_templates = [x.strip() for x in question_templates]


def parseString(input, to_replace, replacements):
    replacements = [int(x) if isinstance(x, float) else x for x in replacements]
    replacements = [str(x) for x in replacements]
    for i, value in enumerate(to_replace):
        input = input.replace("["+value+"]", replacements[i])
    return input.lower().rstrip("?")


def savefile(data, name):
    with open(name, 'w') as f:
        for item in data:
            f.write('%s\n' % item)


def generateQCA():
    for i in range(0,5):
        question_replacements = ["player", "year"]
        answer_replacements = ["player", "year", "points"]
        newDF = season_stats[season_stats['Player'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["Player"]
            player = player.replace("*", "")
            year = row.iloc[0]["Year"]
            points = row.iloc[0]["PTS"]
            tempQ = parseString(question_templates[i], question_replacements, [player, year])
            tempA = parseString(answer_templates[0], answer_replacements, [player, year, points])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(5,10):
        question_replacements = ["player"]
        answer_replacements = ["player", "team"]
        print(question_templates[i])
        newDF = season_stats[season_stats['Player'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["Player"]
            player = player.replace("*", "")
            team = row.iloc[0]["Tm"]
            tempQ = parseString(question_templates[i], question_replacements, [player])
            tempA = parseString(answer_templates[1], answer_replacements, [player, team])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(10,15):
        question_replacements = ["award", "year"]
        answer_replacements = ["player", "award", "year"]
        print(question_templates[i])
        newDF = playerAwards_stats[playerAwards_stats['playerID'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["playerID"]
            award = row.iloc[0]["award"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [award, year])
            tempA = parseString(answer_templates[2], answer_replacements, [player, award, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(15, 20):
        question_replacements = ["award", "year"]
        answer_replacements = ["coach", "award", "year"]
        print(question_templates[i])
        newDF = coach_stats[coach_stats['coachID'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["coachID"]
            award = row.iloc[0]["award"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [award, year])
            tempA = parseString(answer_templates[3], answer_replacements, [player, award, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(20, 25):
        question_replacements = ["year"]
        answer_replacements = ["team", "year"]
        print(question_templates[i])
        newDF = champion_stats[champion_stats['tmIDWinner'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            team = row.iloc[0]["tmIDWinner"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [year])
            tempA = parseString(answer_templates[4], answer_replacements, [team, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(25, 30):
        question_replacements = ["game", "year"]
        answer_replacements = ["winTeam", "loseTeam", "game", "year"]
        print(question_templates[i])
        newDF = playoff_stats[playoff_stats['tmIDLoser'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            winteam = row.iloc[0]["tmIDWinner"]
            loseteam = row.iloc[0]["tmIDLoser"]
            game = row.iloc[0]["round"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [game, year])
            tempA = parseString(answer_templates[5], answer_replacements, [winteam, loseteam, game, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(30, 35):
        question_replacements = ["player"]
        answer_replacements = ["player", "year"]
        print(question_templates[i])
        newDF = hof_stats[hof_stats['name'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["name"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [player])
            tempA = parseString(answer_templates[6], answer_replacements, [player, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(35, 40):
        question_replacements = ["player"]
        answer_replacements = ["player", "draftNumber", "year"]
        print(question_templates[i])
        newDF = draft_stats[draft_stats['firstName'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["firstName"] + " " + row.iloc[0]["lastName"]
            year = row.iloc[0]["draftYear"]
            draftNumber = row.iloc[0]["draftOverall"]
            tempQ = parseString(question_templates[i], question_replacements, [player])
            tempA = parseString(answer_templates[7], answer_replacements, [player, draftNumber, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(40, 45):
        question_replacements = ["player"]
        answer_replacements = ["player", "school"]
        print(question_templates[i])
        newDF = draft_stats[draft_stats['draftFrom'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            player = row.iloc[0]["firstName"] + " " + row.iloc[0]["lastName"]
            school = row.iloc[0]["draftFrom"]
            tempQ = parseString(question_templates[i], question_replacements, [player])
            tempA = parseString(answer_templates[8], answer_replacements, [player, school])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)
    for i in range(45, 50):
        question_replacements = ["team", "time"]
        answer_replacements = ["team", "wins", "loses", "time"]
        print(question_templates[i])
        newDF = team_stats[team_stats['name'].notna()]
        newDF.reset_index()
        leng = len(newDF.index)
        for j in range(0, 50):
            index = random.randrange(0, leng)
            row = newDF.iloc[[index]]
            row.reset_index()
            team = row.iloc[0]["name"]
            wins = row.iloc[0]["won"]
            loses = row.iloc[0]["lost"]
            year = row.iloc[0]["year"]
            tempQ = parseString(question_templates[i], question_replacements, [team, year])
            tempA = parseString(answer_templates[9], answer_replacements, [team, wins, loses, year])
            tempC = row.to_string(header=False)
            questions_final.append(tempQ)
            answers_final.append(tempA)
            context_final.append(tempC)



generateQCA()
print(parseString("This is a [test] string?", ["test"], ["Replacement"]))
savefile(questions_final, "questionsFinal.txt")
savefile(answers_final, "answersFinal.txt")
savefile(context_final, "contextFinal.txt")
