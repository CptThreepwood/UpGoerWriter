import dotenv

from os.path import join, dirname

dotenv.load_dotenv(join(dirname(__file__), '.env'))

from .ModelLoader import Translator