import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt

# step 1

def open_file(file, kills = True):
    """Opens file, then strips and splits the elements of the file. 
    Converts values in the date column into datetime format.
    Returns a structured array.
    
    Parameters:
    - file: .txt file;
    - kills: boolean, True if file contains kills per match,
    False if it contains player ids of cheaters."""

    if kills:
        data = np.genfromtxt(file, names = ['match_id', 'killer_id', 'victim_id', 'date_time'],
                                    dtype = ['U36', 'U40', 'U40', 'M8[ms]'],
                                    delimiter = "\t")

    else:
        data = np.genfromtxt(file, dtype = {'names':('cheater_id', 'date_cheating', 'date_banned'),
                          'formats':('U40', 'datetime64[D]', 'datetime64[D]')})

    return data

def toy_set(n, data):
    """Returns smaller sample of data array, of length n."""
    toy_set_rw = data[0:n]
    toy_set_lm = toy_set_rw[-1][0]

    toy_set = np.delete(toy_set_rw, np.where(toy_set_rw["match_id"] != toy_set_lm)[0], axis = 0)

    return toy_set

def matches_dates(kills):
    """Returns dictionary with a match id as key
    and the date when the match finished as value.

    Parameters:
        - kills: structured array containing killings in matches
    """

    matches = np.unique(kills["match_id"])

    matches_dates_dict = {match : None for match in matches}

    for match in matches:
        msk_match = kills["match_id"] == match
        kills_tmp = kills[msk_match][:]

        date = kills_tmp[-1]["date_time"]
        date_match = date.astype('datetime64[D]')

        matches_dates_dict[match] = date_match
    
    return matches_dates_dict


def victims_cheating(kills, cheaters):
    """Returns two dictionaries.
    The first dictionary has all players killed by cheating
    with time of kill, per match.
    The second dictonnary has all players killed by cheating,
    per match, without the time of kill.

    Parameters:
        - kills: structured array containing
        killings in matches;
        - cheaters: structured array of cheaters.
    """
    matches = np.unique(kills["match_id"])

    victims_cheating_times_dict = {match : None for match in matches}
    victims_cheating_dict = {match : None for match in matches}

    for match in matches:
        # masks to filter arrays from:
        # https://stackoverflow.com/questions/58079075/numpy-select-rows-based-on-condition
        # https://stackoverflow.com/questions/6792395/how-to-mask-numpy-structured-array-on-multiple-columns

        msk_match = kills["match_id"] == match
        kills_tmp = kills[msk_match][:]

        killers = np.unique(kills_tmp["killer_id"])

        date = kills_tmp[0]["date_time"]
        
        msk_time = cheaters["date_cheating"] <= date
        already_cheater = cheaters[msk_time][:]

        cheat_killers = [killer for killer in killers if killer in already_cheater["cheater_id"]]
    
        cheat_victims_times = []
        cheat_victims = []
        
        for cheater in cheat_killers:
            msk_cheater = kills_tmp["killer_id"] == cheater
            kills_cheat = kills_tmp[msk_cheater][:] 
            cheat_victims_times.append(list(zip(kills_cheat["victim_id"], kills_cheat["date_time"])))
            cheat_victims = (list(kills_cheat["victim_id"]))

        if len(cheat_victims) !=  0:
            victims_cheating_times_dict[match] = cheat_victims_times

            victims_cheating_dict[match] = cheat_victims

    victims_cheating_times_dict = {k: v for k, v in victims_cheating_times_dict.items() if v is not None}
    victims_cheating_dict = {k: v for k, v in victims_cheating_dict.items() if v is not None}

    return victims_cheating_times_dict, victims_cheating_dict


def matches_cheating(victims_cheating_times_dict):
    """Returns a dictionary with the time of earliest
    3rd kill by cheating, per match.

    Parameters:
        - victims_cheating_dict:
        dictionary with all players killed by
        cheating, per match, and time of kill.
    """
    matches_cheating_dict = {}

    for match in victims_cheating_times_dict.keys():
        if len(victims_cheating_times_dict[match]) != 0:

            num_cheaters = len(victims_cheating_times_dict[match])
            times_3rd_kill = []

            for cheater in range(num_cheaters):
                num_cheating_kills = len(victims_cheating_times_dict[match][cheater]) 

                if num_cheating_kills > 3:
                    times_3rd_kill.append(victims_cheating_times_dict[match][cheater][2][1])
                    matches_cheating_dict[match] = min(times_3rd_kill)

                else:
                    pass
        
    return matches_cheating_dict


def witness_cheating(matches_cheating_dict, kills):
    """Returns a dictionary with all players that observed
    cheating and the time of earliest 3rd kill by cheating,
    per match.
    
    Parameters:
        - matches_cheating_dict:
        dictionary containing a match id as key and the time
        of earliest 3rd kill by cheating of that match as value;
        - kills: structured array containing killings in matches.
    """
    witness_cheating_dict = {}

    for match in matches_cheating_dict.keys():
        all_players = []
        kills_tmp = kills[kills["match_id"] == match]
            
        all_players = list(kills_tmp["killer_id"]) + list(kills_tmp["victim_id"])
        unique_players = list(set(all_players))

        time_3rd_kill = matches_cheating_dict[match]

        msk_time = kills_tmp["date_time"] < time_3rd_kill
        time_cheater = kills_tmp[msk_time]
        all_players_dead = list(time_cheater["victim_id"])

        all_witness = [obs for obs in unique_players if obs not in all_players_dead]

        witness_cheating_dict[match] = [all_witness, time_3rd_kill]  

    return witness_cheating_dict


def victims_cheaters(victims_cheating_dict, matches_dates_dict, cheaters):
    """Returns a list with all players that were victims
    of cheating and started cheating whitin 5 days.

    Parameters:
    – victims_cheating_dict: dictionary with all players that observed
    cheating and the time of earliest 3rd kill by cheating, per match.
    """
    victims_cheaters_list = []

    for match in victims_cheating_dict.keys():
        date = matches_dates_dict[match]
        five_days = np.timedelta64(5, 'D')
        five_days_later = date + five_days
        
        msk_time = cheaters["date_cheating"] > date
        cheaters_after_match = cheaters[msk_time]

        msk_time2 = cheaters_after_match["date_cheating"] < five_days_later
        cheaters_five_days_arr = cheaters_after_match[msk_time2]
        cheaters_five_days = list(cheaters_five_days_arr["cheater_id"])


        for victim in victims_cheating_dict[match]:
            if victim in cheaters_five_days:
                victims_cheaters_list.append(victim)


    return victims_cheaters_list


def witness_cheaters(witness_cheating_dict, matches_dates_dict, cheaters):
    """Returns a list with all players that witnessed cheating
    and started cheating within 5 days.

    Parameters:
    – witness_cheating_dict: dictionary with all players that observed
    cheating and the time of earliest 3rd kill by cheating, per match;
    - matches_dates_dict: a dictionary with the time of earliest
    3rd kill by cheating, per match;
    - cheaters: structured array of cheaters.
    """

    witness_cheaters_list = []

    for match in witness_cheating_dict.keys():
        date = matches_dates_dict[match]
        five_days = np.timedelta64(5, 'D')
        five_days_later = date + five_days
        
        msk_time = cheaters["date_cheating"] > date
        cheaters_after_match = cheaters[msk_time]

        msk_time2 = cheaters_after_match["date_cheating"] < five_days_later
        cheaters_five_days_arr = cheaters_after_match[msk_time2]
        cheaters_five_days = list(cheaters_five_days_arr["cheater_id"])

        witness_list = witness_cheating_dict[match]

        for witness in witness_list[0]:
            if witness in cheaters_five_days:
                witness_cheaters_list.append(witness)

    return witness_cheaters_list


def observers_cheaters_results(victims_cheaters_list, witness_cheaters_list, detail = False):
    """Returns a list of all players that observed cheating and
    started cheating within 5 days, and an integer with the length
    of that list.

    Parameters:
    – victims_cheaters_list: list with all players that were victims
    of cheating and started cheating whitin 5 days;
    - witness_cheaters_list: ist with all players that witnessed cheating
    and started cheating within 5 days;
    - detail: a boolean to select results broken down between those
    players that were killed by cheating and those that witnessed cheating.
    """

    observers_cheaters_list = victims_cheaters_list + witness_cheaters_list
    num_observers = len(observers_cheaters_list)
    num_victims = len(victims_cheaters_list)
    num_witness = len(witness_cheaters_list)

    if detail == False:
        print("There are", num_observers, "players that started cheating within 5 days of observing cheating.")
    else:
        print("- There are", num_observers, "players that started cheating within 5 days of observing cheating.")
        print("- From those", num_observers, "players,", num_victims, "were killed by a cheater and", num_witness, "witnessed cheating.")
        print("- Here is the list of players who started cheating within 5 days of observing cheating:")
        print("  ")
        for cheater in observers_cheaters_list:
            print(cheater)
    
    return observers_cheaters_list, num_observers


def how_many_cheaters(kills, cheaters, detail = False):
    """Wrap up function. Returns an integer with the total number
    of players that observed cheating and started cheating within 5 days.

    Parameters:
    - kills: structured array containing
        killings in matches;
    - cheaters: structured array of cheaters;
    - detail: a boolean to select results broken down between those
    players that were killed by cheating and those that witnessed cheating.
    """

    matches_dates_dict = matches_dates(kills)

    victims_cheating_times_dict, victims_cheating_dict = victims_cheating(kills, cheaters)

    matches_cheating_dict = matches_cheating(victims_cheating_times_dict)

    witness_cheating_dict = witness_cheating(matches_cheating_dict, kills)

    victims_cheaters_list = victims_cheaters(victims_cheating_dict, matches_dates_dict, cheaters)

    witness_cheaters_list = witness_cheaters(witness_cheating_dict, matches_dates_dict, cheaters)

    observers_cheaters_list, num_observers = observers_cheaters_results(victims_cheaters_list, witness_cheaters_list, detail)

    return num_observers


# step 2

def simulation(kills, cheaters):
    """Returns a copy of the structured array kills, with all players
    but the cheater randomized within each match.

    Parameters:
    - kills: structured array containing
    killings in matches;
    - cheaters: structured array of cheaters.
    """

    matches = np.unique(kills["match_id"])

    kills_alternative = []

    for match in matches:
        kills_tmp = kills[kills["match_id"] == match]

        kills_altern_match = kills_tmp

        date = kills_tmp[0]["date_time"]
        
        already_cheater = list(cheaters["cheater_id"][cheaters["date_cheating"] <= date])

        killers = list(np.unique(kills_tmp["killer_id"]))
        victims = list(np.unique(kills_tmp["victim_id"]))

        cheat_killers = [killer for killer in killers if killer in already_cheater]

        players = list(set(killers + victims))

        if len(cheat_killers) == 0:
            kills_alternative.append(kills_altern_match)

        else:
            for cheater in cheat_killers:
                players.remove(cheater)
            num_players = len(players)

            list_index_kill = []
            list_index_vict = []

            for player in players:
                index_killer_arr = np.where(kills_tmp["killer_id"] == player)
                index_killer = index_killer_arr[0].tolist()
                list_index_kill.append(index_killer)

                index_victim_arr = np.where(kills_tmp["victim_id"] == player)
                index_victim = index_victim_arr[0].tolist()
                list_index_vict.append(index_victim)


            players_shuffled = random.sample(players, num_players)

            for index in range(num_players):
                if len(list_index_kill[index]) == 0:
                    pass
                else:
                    for i in list_index_kill[index]:
                        kills_altern_match["killer_id"][i] = players_shuffled[index]

                if len(list_index_vict[index]) == 0:
                    pass
                else:
                    for i in list_index_vict[index]:
                        kills_altern_match["victim_id"][i] = players_shuffled[index]
            

            kills_alternative.append(kills_altern_match)

    kills_alternative = np.concatenate(kills_alternative)
    

    return kills_alternative

