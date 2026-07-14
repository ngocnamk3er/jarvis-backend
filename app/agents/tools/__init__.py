from app.agents.tools.bash import bash
from app.agents.tools.web_search import web_search
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.represent_file import represent_file
from app.agents.tools.generate_visualization_svg import generate_visualization_svg

tools = [bash, web_search, web_fetch, represent_file, generate_visualization_svg]
