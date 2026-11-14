from app.agents.tools.ask_user import ask_user
from app.agents.tools.bash import bash
from app.agents.tools.check_pdf import check_pdf
from app.agents.tools.files import fetch_file, list_files, read_file, search_files
from app.agents.tools.generate_visualization_svg import generate_visualization_svg
from app.agents.tools.present_file import present_file
from app.agents.tools.view_image import view_image
from app.agents.tools.ocr_tools import draw_boxes, ocr_image
from app.agents.tools.office_tools import office_convert, office_recalculate
from app.agents.tools.read_pdf import read_pdf
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
    search_files,
    fetch_file,
    view_image,
    check_pdf,
    read_pdf,
    office_convert,
    office_recalculate,
    ocr_image,
    draw_boxes,
]
