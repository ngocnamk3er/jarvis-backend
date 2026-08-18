from app.agents.tools.web_search import web_search
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.generate_visualization_svg import generate_visualization_svg
from app.agents.tools.ask_user import ask_user

tools = [web_search, web_fetch, generate_visualization_svg, ask_user]
