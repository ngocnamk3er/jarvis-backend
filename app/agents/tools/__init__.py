from app.agents.tools.get_current_time import get_current_time
from app.agents.tools.calculator import calculator
from app.agents.tools.bash import bash
from app.agents.tools.web_search import web_search
from app.agents.tools.represent_file import represent_file
from app.agents.tools.generate_visualization_mermaid import generate_visualization_mermaid
from app.agents.tools.generate_visualization_svg import generate_visualization_svg
from app.agents.tools.generate_animation import generate_animation

tools = [get_current_time, calculator, bash, web_search, represent_file, generate_visualization_mermaid, generate_visualization_svg, generate_animation]
