from app.agents.tools.ask_user import ask_user
from app.agents.tools.bash import bash
from app.agents.tools.files import grep_files, list_files, read_file, search_files
from app.agents.tools.generate_visualization_svg import generate_visualization_svg
from app.agents.tools.present_file import present_file
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.web_search import web_search

tools = [
    bash,
    present_file,
    web_search,
    web_fetch,
    generate_visualization_svg,
    ask_user,
    list_files,
    read_file,
    grep_files,
    search_files,
]
