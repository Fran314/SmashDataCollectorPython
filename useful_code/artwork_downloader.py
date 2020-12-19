import requests
import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2
import re


CHARACTER_NAMES = [#["MARIO",                    "Mario"],
                    # ["DONKEY KONG",             "Donkey Kong"],
                    # ["LINK",                    "Link"],
                    # ["SAMUS",                   "Samus"],
                    # ["SAMUS OSCURA",            "Dark Samus"],
                    # ["YOSHI",                   "Yoshi"],
                    # ["KIRBY",                   "Kirby"],
                    # ["FOX",                     "Fox"],
                    # ["PIKACHU",                 "Pikachu"],
                    # ["LUIGI",                   "Luigi"],
                    # ["NESS",                    "Ness"],
                    # ["CAPTAIN FALCON",          "Captain Falcon"],
                    # ["JIGGLYPUFF",              "Jigglypuff"],
                    # ["PEACH",                   "Peach"],
                    # ["DAISY",                   "Daisy"],
                    # ["BOWSER",                  "Bowser"],
                    # ["ICE CLIMBERS",            "Ice Climbers"],
                    # ["SHEIK",                   "Sheik"],
                    # ["ZELDA",                   "Zelda"],
                    # ["DR. MARIO",               "Dr. Mario"],
                    # ["PICHU",                   "Pichu"],
                    # ["FALCO",                   "Falco"],
                    # ["MARTH",                   "Marth"],
                    # ["LUCINA",                  "Lucina"],
                    # ["LINK BAMBINO",            "Young Link"],
                    # ["GANONDORF",               "Ganondorf"],
                    # ["MEWTWO",                  "Mewtwo"],
                    # ["ROY",                     "Roy"],
                    # ["CHROM",                   "Chrom"],
                    # ["MR. GAME & WATCH",        "Mr. Game & Watch"],
                    # ["META KNIGHT",             "Meta Knight"],
                    # ["PIT",                     "Pit"],
                    # ["PIT OSCURO",              "Dark Pit"],
                    # ["SAMUS TUTA ZERO",         "Zero Suit Samus"],
                    # ["WARIO",                   "Wario"],
                    # ["SNAKE",                   "Snake"],
                    # ["IKE",                     "Ike"],
                    # ["ALLENATORE DI POKÉMON",   "Pokémon Trainer"],
                    # ["DIDDY KONG",              "Diddy Kong"],
                    # ["LUCAS",                   "Lucas"],
                    # ["SONIC",                   "Sonic"],
                    # ["KING DEDEDE",             "King Dedede"],
                    # ["OLIMAR",                  "Olimar"],
                    # ["ALPH",                    "Olimar"],
                    # ["LUCARIO",                 "Lucario"],
                    # ["R.O.B.",                  "R.O.B."],
                    # ["LINK CARTONE",            "Toon Link"],
                    # ["WOLF",                    "Wolf"],
                    # ["ABITANTE",                "Villager"],
                    # ["MEGA MAN",                "Mega Man"],
                    # ["TRAINER DI WII FIT",      "Wii Fit Trainer"],
                    # ["ROSALINDA E SFAVILLOTTO", "Rosalina & Luma"],
                    # ["LITTLE MAC",              "Little Mac"],
                    # ["GRENINJA",                "Greninja"],
                    # ["PALUTENA",                "Palutena"],
                    # ["PAC-MAN",                 "Pac-Man"],
                    # ["DARAEN",                  "Robin"],
                    # ["SHULK",                   "Shulk"],
                    # ["BOWSER JUNIOR",           "Bowser Jr."],
                    # ["LARRY",                   "Bowser Jr."],
                    # ["ROY",                     "Bowser Jr."],
                    # ["WENDY",                   "Bowser Jr."],
                    # ["IGGY",                    "Bowser Jr."],
                    # ["MORTON",                  "Bowser Jr."],
                    # ["LEMMY",                   "Bowser Jr."],
                    # ["LUDWIG",                  "Bowser Jr."],
                    # ["DUO DUCK HUNT",           "Duck Hunt"],
                    # ["RYU",                     "Ryu"],
                    # ["KEN",                     "Ken"],
                    # ["CLOUD",                   "Cloud"],
                    # ["CORRIN",                  "Corrin"],
                    # ["BAYONETTA",               "Bayonetta"],
                    # ["RAGAZZO INKLING",         "Inkling"],
                    # ["RAGAZZA INKLING",         "Inkling"],
                    # ["RIDLEY",                  "Ridley"],
                    # ["SIMON",                   "Simon"],
                    # ["RICHTER",                 "Richter"],
                    # ["KING K. ROOL",            "King K. Rool"],
                    # ["FUFFI",                   "Isabelle"],
                    # ["INCINEROAR",              "Incineroar"],
                    # ["PIANTA PIRANHA",          "Piranha Plant"],
                    # ["JOKER",                   "Joker"],
                    ["EROE",                    "Dq Hero"],
                    ["BANJO E KAZOOIE",         "Banjo & Kazooie"],
                    ["TERRY",                   "Terry"],
                    ["BYLETH",                  "Byleth"],
                    ["MIN MIN",                 "MinMin"],
                    ["STEVE",                   "Steve"],
                    ["SEPHIROTH",               "Sephiroth"]]
                    # ["ALEX",                    "Steve"],
                    # ["ZOMBIE",                  "Steve"],
                    # ["ENDERMAN",                "Steve"]]

CHARACTER_NAMES = [#["Mario",                "mario"],
                    # ["Donkey Kong",         "donkey_kong"],
                    # ["Link",                "link"],
                    # ["Samus",               "samus"],
                    # ["Dark Samus",          "dark_samus"],
                    # ["Yoshi",               "yoshi"],
                    # ["Kirby",               "kirby"],
                    # ["Fox",                 "fox"],
                    # ["Pikachu",             "pikachu"],
                    # ["Luigi",               "luigi"],
                    # ["Ness",                "ness"],
                    # ["Captain Falcon",      "captain_falcon"],
                    # ["Jigglypuff",          "jigglypuff"],
                    # ["Peach",               "peach"],
                    # ["Daisy",               "daisy"],
                    # ["Bowser",              "bowser"],
                    # ["Ice Climbers",        "ice_climbers"],
                    # ["Sheik",               "sheik"],
                    # ["Zelda",               "zelda"],
                    # ["Dr. Mario",           "dr_mario"],
                    # ["Pichu",               "pichu"],
                    # ["Falco",               "falco"],
                    # ["Marth",               "marth"],
                    # ["Lucina",              "lucina"],
                    # ["Young Link",          "young_link"],
                    # ["Ganondorf",           "ganondorf"],
                    # ["Mewtwo",              "mewtwo"],
                    # ["Roy",                 "roy"],
                    # ["Chrom",               "Chrom"],
                    # ["Mr. Game & Watch",    "mr_game_and_watch"],
                    # ["Meta Knight",         "meta_knight"],
                    # ["Pit",                 "pit"],
                    # ["Dark Pit",            "dark_pit"],
                    # ["Zero Suit Samus",     "zero_suit_samus"],
                    # ["Wario",               "wario"],
                    # ["Snake",               "snake"],
                    # ["Ike",                 "ike"],
                    # ["Pokémon Trainer",     "pokemon_trainer"],
                    # ["Diddy Kong",          "diddy_kong"],
                    # ["Lucas",               "lucas"],
                    # ["Sonic",               "sonic"],
                    # ["King Dedede",         "king_dedede"],
                    # ["Olimar",              "olimar"],
                    # ["Lucario",             "lucario"],
                    # ["R.O.B.",              "rob"],
                    # ["Toon Link",           "toon_link"],
                    # ["Wolf",                "wolf"],
                    # ["Villager",            "villager"],
                    # ["Mega Man",            "mega_man"],
                    # ["Wii Fit Trainer",     "wii_fit_trainer"],
                    # ["Rosalina & Luma",     "rosalina_and_luma"],
                    # ["Little Mac",          "little_mac"],
                    # ["Greninja",            "greninja"],
                    # ["Palutena",            "palutena"],
                    # ["Pac-Man",             "pac_man"],
                    # ["Robin",               "robin"],
                    # ["Shulk",               "shulk"],
                    # ["Bowser Jr.",          "bowser_jr"],
                    # ["Duck Hunt",           "duck_hunt"],
                    # ["Ryu",                 "ryu"],
                    # ["Ken",                 "ken"],
                    # ["Cloud",               "cloud"],
                    # ["Corrin",              "corrin"],
                    # ["Bayonetta",           "bayonetta"],
                    # ["Inkling",             "inkling"],
                    # ["Ridley",              "ridley"],
                    # ["Simon",               "simon"],
                    # ["Richter",             "richter"],
                    # ["King K. Rool",        "king_k_rool"],
                    # ["Isabelle",            "isabelle"],
                    # ["Incineroar",          "incineroar"],
                    # ["Piranha Plant",       "piranha_plant"],
                    # ["Joker",               "joker"],
                    # ["Hero",                "dq_hero"],
                    # ["Banjo & Kazooie",     "banjo_and_kazooie"],
                    # ["Terry",               "terry"],
                    # ["Byleth",              "byleth"],
                    # ["Min Min",             "minmin"],
                    # ["Steve",               "steve"],
                    ["Sephiroth",           "sephiroth"]]


def folderizeName(arg):
    to_return = ""
    for c in arg:
        if(ord(c) >= 97 and ord(c) <= 122):
            to_return += c
        elif(ord(c) >= 65 and ord(c) <= 90):
            to_return += chr(ord(c) + 32)
        elif(c == ' ' or c =='-'):
            to_return += '_'
        elif(c == 'é'):
            to_return += 'e'
        elif(c == '&'):
            to_return += "and"
    return to_return

def int2str(arg):
    if(arg == 1):
        return ""
    else:
        return str(arg)

for name in CHARACTER_NAMES:
    character_path = os.path.join(r'D:\Utente\Desktop\smash\clean_artworks', name[1])
    images = []
    for i in range(8):
        img_data = requests.get(r'https://www.smashbros.com/assets_v2/img/fighter/' + name[1] + r'/main' + int2str(i+1) + r'.png').content
        handler = open(os.path.join(character_path, str(i) + ".png"), 'wb')
        handler.write(img_data)
        handler.close()
        images.append(cv2.imread(os.path.join(character_path, str(i) + ".png")))

    shape = images[0].shape
    for i in range(1,8):
        if(images[i].shape != shape):
            print(f'Error at {name[1]}')