# import libraries
from urllib.request import urlopen
import requests
import os
import pandas as pd
import numpy as np
import re
import itertools
import json

from pubchemScrapper import hasExperimentalProperties, getFlashpoint, getCompoundName, getSmiles
from pubchemScrapper import getClosedCupFlashpoint, strToNegInt, findUnit, findTemp, farnenheitToCelsius