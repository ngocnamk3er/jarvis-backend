from app.agents.tools.get_current_time import get_current_time
from app.agents.tools.bash import bash
from app.agents.tools.web_search import web_search
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.read_skill import read_skill
from app.agents.tools.represent_file import represent_file
from app.agents.tools.generate_animation import generate_animation
from app.agents.tools.generate_webapp import generate_webapp

tools = [get_current_time, bash, web_search, web_fetch, read_skill, represent_file, generate_animation, generate_webapp]
