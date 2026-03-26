import pathlib
import textwrap
import os
import PIL.Image
import json
import argparse
# import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from .utils.decision_generation import decision_generation
from .utils.split_task import split_task
from .utils.execute_modularization import execute_modularization, summary
from .utils.execute_synthesis import execute_synthesis

def cantor_forward(query, img_path, model):
    decision = {}
    decision = decision_generation(query, img_path, model, decision)
    decision = split_task(query, img_path, model, decision)
    decision = execute_modularization(query, img_path, model, decision)
    decision = summary(query, img_path, model, decision)
    decision = execute_synthesis(query, img_path, model, decision)
    return decision