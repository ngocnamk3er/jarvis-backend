"""ocr_image / draw_boxes: boxes come from OCR, never from coordinates the model chooses."""

import importlib
import io
import json
import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.agents import ocr

tools = importlib.import_module("app.agents.tools.ocr_tools")

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def row(block, par, line, word, left, top, width, height, conf, text):
    return f"5\t1\t{block}\t{par}\t{line}\t{word}\t{left}\t{top}\t{width}\t{height}\t{conf}\t{text}"


def png(size=(300, 200), color="white") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


# ---- reading tesseract's output ----------------------------------------------------------------

def test_words_are_grouped_into_lines_ordered_top_to_bottom_and_scaled_back():
    tsv = "\n".join([HEADER,
        row(1, 1, 2, 1, 40, 300, 100, 20, 90, "Second"), row(1, 1, 2, 2, 150, 300, 80, 20, 90, "line"),
        row(1, 1, 1, 2, 160, 100, 60, 20, 95, "world"), row(1, 1, 1, 1, 40, 100, 100, 20, 95, "Hello"),
        row(1, 1, 3, 1, 10, 500, 50, 20, -1, "noise"), row(1, 1, 3, 2, 10, 500, 50, 20, 80, "  ")])
    lines = ocr.parse_tsv(tsv, scale=2)
    assert [l.text for l in lines] == ["Hello world", "Second line"] and [l.id for l in lines] == [1, 2]
    assert lines[0].box == (20, 50, 110, 60)                      # divided by the 2x enlargement
    assert [w.text for w in lines[0].words] == ["Hello", "world"]   # words left to right


def test_matching_ignores_diacritics_and_survives_misreadings():
    line = ocr.parse_tsv("\n".join([HEADER,
        row(1, 1, 1, 1, 0, 0, 40, 10, 90, "Mi"), row(1, 1, 1, 2, 50, 0, 40, 10, 90, "số"),
        row(1, 1, 1, 3, 100, 0, 60, 10, 90, "thuế"), row(1, 1, 1, 4, 170, 0, 80, 10, 90, "code):"),
        row(1, 1, 1, 5, 260, 0, 140, 10, 90, "0309405347")]), 1)[0]
    box, matched, score = ocr.locate(line, "0309405347")
    assert box == (260, 0, 400, 10) and matched == "0309405347" and score == 1.0
    box, matched, _ = ocr.locate(line, "Mã số thuế")                  # 'Mi' for 'Mã' is close enough
    assert box[0] == 0 and matched.startswith("Mi số thuế")
    assert ocr.locate(line, "completely different words") is None


def test_numbers_are_not_matched_loosely_to_other_numbers():
    line = ocr.parse_tsv("\n".join([HEADER, row(1, 1, 1, 1, 0, 0, 80, 10, 90, "1.320.000")]), 1)[0]
    assert ocr.locate(line, "1.320.000") is not None
    assert ocr.locate(line, "9.876.543") is None


# ---- drawing ------------------------------------------------------------------------------------

def test_drawing_frames_the_value_and_the_key_in_one_colour_and_adds_nothing_around_the_picture():
    out = Image.open(io.BytesIO(ocr.draw_boxes(png((300, 200)), [("Total", (150, 80, 250, 100), (50, 80, 120, 100))]))).convert("RGB")
    assert out.size == (300, 200)                                     # no legend strip, no margin
    assert out.getpixel((149, 90)) != (255, 255, 255)                 # the value's frame (left edge)
    assert out.getpixel((48, 90)) != (255, 255, 255)                  # the key's frame (left edge, a thinner line)
    assert out.getpixel((149, 90)) == out.getpixel((48, 90))          # in the same colour
    assert out.getpixel((200, 90)) == (255, 255, 255) and out.getpixel((80, 90)) == (255, 255, 255)   # inside both is left clear
    assert out.getpixel((5, 5)) == (255, 255, 255)                    # nothing drawn elsewhere
    assert all(out.getpixel((x, y)) == (255, 255, 255) for x in range(40, 260, 5) for y in (70, 71))   # no label: the key says what it is


def test_a_field_with_no_key_frame_has_its_label_written_beside_the_value():
    out = Image.open(io.BytesIO(ocr.draw_boxes(png((300, 200)), [("Total", (150, 80, 250, 100), None)]))).convert("RGB")
    assert out.size == (300, 200)
    assert any(out.getpixel((x, y)) != (255, 255, 255) for x in range(150, 250) for y in range(60, 80))   # the tag above the box


# ---- the tools, against a fake sandbox and a fake OCR ----------------------------------------------

LINES = [ocr.Line(1, "Mã số thuế: 0309405347", (20, 40, 220, 58),
                  [ocr.Word("Mã", (20, 40, 40, 58)), ocr.Word("số", (46, 40, 66, 58)), ocr.Word("thuế:", (72, 40, 120, 58)),
                   ocr.Word("0309405347", (130, 40, 220, 58))]),
         ocr.Line(2, "Tổng cộng 1.320.000", (20, 100, 260, 118),
                  [ocr.Word("Tổng", (20, 100, 70, 118)), ocr.Word("cộng", (76, 100, 126, 118)), ocr.Word("1.320.000", (180, 100, 260, 118))])]


def _cell(r, c, box, text, span=(1, 1)):
    words = [ocr.Word(t, (box[0] + 4 + i * 40, box[1] + 4, box[0] + 40 + i * 40, box[3] - 4)) for i, t in enumerate(text.split())]
    return ocr.Cell(r, c, box, words, span[0], span[1])


TABLES = [ocr.Table(1, (10, 130, 290, 190), 2, 3, [
    _cell(1, 1, (10, 130, 60, 160), "STT"), _cell(1, 2, (60, 130, 200, 160), "Tên hàng"), _cell(1, 3, (200, 130, 290, 160), "Thành tiền"),
    _cell(2, 1, (10, 160, 60, 190), "1"), _cell(2, 2, (60, 160, 200, 190), "Phần mềm"), _cell(2, 3, (200, 160, 290, 190), "1.320.000")])]


@pytest.fixture
def sandbox(monkeypatch):
    files = {"inv.png": png((300, 200))}

    async def read_file(thread_id, name):
        if name not in files:
            raise FileNotFoundError(name)
        return files[name], "image/png", name

    async def write_file(thread_id, name, content):
        files[name] = content

    monkeypatch.setattr(tools, "read_file", read_file)
    monkeypatch.setattr(tools, "write_file", write_file)
    monkeypatch.setattr(ocr, "run_ocr", lambda data, lang=None, psm=None: ((300, 200), LINES, TABLES))
    return files


async def call(tool, **args):
    return await tool.ainvoke({"label": "x", **args}, config={"configurable": {"thread_id": "t"}})


async def test_ocr_lists_every_line_with_its_id_and_box_and_saves_the_result(sandbox):
    out = await call(tools.ocr_image, path="inv.png")
    assert "L1 [20,40,220,58] Mã số thuế: 0309405347" in out and "L2 [20,100,260,118] Tổng cộng 1.320.000" in out
    assert "Do not write coordinates yourself" in out
    assert "for finding POSITIONS" in out and "view_image" in out                    # the text is for positions, values come from the picture
    saved = json.loads(sandbox["inv.png.ocr.json"])
    assert saved["sha256"] == ocr.sha256(sandbox["inv.png"]) and saved["lines"][0]["words"][3]["text"] == "0309405347"


async def test_the_drawn_picture_reaches_the_chat_without_the_model_presenting_it(sandbox):
    from langchain_core.messages import AIMessage, ToolMessage

    from app.services.chat_service import ToolEndEventHandler
    from app.services.conversation_service import serialize_messages

    await call(tools.ocr_image, path="inv.png")
    msg = await tools.draw_boxes.ainvoke(
        {"name": "draw_boxes", "id": "c1", "type": "tool_call", "args": {"label": "x", "path": "inv.png", "items": [{"label": "Tax", "lines": [1]}]}},
        config={"configurable": {"thread_id": "t"}})
    assert isinstance(msg, ToolMessage)
    chip = {"name": "inv_boxes.png", "mime": "image/png", "size": len(sandbox["inv_boxes.png"]), "path": "inv_boxes.png", "thread_id": "t"}
    assert msg.artifact == {"files": [chip]}

    # while streaming: the step, then the file chip present_file would have made
    events = ToolEndEventHandler().handle({"name": "draw_boxes", "run_id": "r1", "data": {"output": msg}})
    assert [e["type"] for e in events] == ["tool_end", "file"] and events[1]["path"] == "inv_boxes.png"
    # after a reload: the same chip, after the step
    ai = AIMessage(content="", tool_calls=[{"name": "draw_boxes", "id": "c1", "args": {"path": "inv.png", "items": []}}])
    parts = serialize_messages([ai, msg])[0]["parts"]
    assert [p["type"] for p in parts] == ["tool", "file"] and parts[1]["name"] == "inv_boxes.png"


async def test_nothing_is_attached_when_nothing_was_drawn(sandbox):
    msg = await tools.draw_boxes.ainvoke(
        {"name": "draw_boxes", "id": "c2", "type": "tool_call", "args": {"label": "x", "path": "inv.png", "items": [{"label": "x", "lines": [1]}]}},
        config={"configurable": {"thread_id": "t"}})
    assert msg.content.startswith("Error: there is no OCR result") and msg.artifact is None


async def test_the_key_gets_a_box_of_its_own_beside_the_value(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[
        {"label": "Mã số thuế", "lines": [1], "text": "0309405347", "key": "Mã số thuế:"}])
    record = json.loads(sandbox["inv_boxes.json"])["boxes"][0]
    assert record["box"] == [130, 40, 220, 58] and record["key_box"] == [20, 40, 120, 58]      # two boxes, side by side
    assert "key [20,40,120,58]" in out and "value [130,40,220,58]" in out
    img = Image.open(io.BytesIO(sandbox["inv_boxes.png"])).convert("RGB")
    assert img.size == (300, 200)                                                    # no legend under it
    assert img.getpixel((18, 49)) != (255, 255, 255) and img.getpixel((128, 49)) != (255, 255, 255)   # a frame on each


async def test_a_key_that_is_not_on_the_picture_is_reported_and_only_the_value_is_boxed(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[
        {"label": "Mã số thuế", "lines": [1], "text": "0309405347", "key": "Số tài khoản ngân hàng"}])
    record = json.loads(sandbox["inv_boxes.json"])["boxes"][0]
    assert record["key_box"] is None and record["box"] == [130, 40, 220, 58]
    assert "the key 'Số tài khoản ngân hàng' was not found" in out


async def test_a_key_printed_twice_is_taken_from_beside_the_value_asked_for(sandbox, monkeypatch):
    twice = [ocr.Line(1, "Tên đơn vị: Alpha", (20, 20, 200, 38),
                      [ocr.Word("Tên", (20, 20, 45, 38)), ocr.Word("đơn", (50, 20, 80, 38)), ocr.Word("vị:", (85, 20, 105, 38)),
                       ocr.Word("Alpha", (130, 20, 200, 38))]),
             ocr.Line(2, "Tên đơn vị: Beta", (20, 120, 190, 138),
                      [ocr.Word("Tên", (20, 120, 45, 138)), ocr.Word("đơn", (50, 120, 80, 138)), ocr.Word("vị:", (85, 120, 105, 138)),
                       ocr.Word("Beta", (130, 120, 190, 138))])]
    monkeypatch.setattr(ocr, "run_ocr", lambda data, lang=None, psm=None: ((300, 200), twice, []))
    await call(tools.ocr_image, path="inv.png")
    await call(tools.draw_boxes, path="inv.png", items=[{"label": "Buyer", "lines": [2], "text": "Beta", "key": "Tên đơn vị:"},
                                                         {"label": "Seller", "lines": [1], "text": "Alpha", "key": "Tên đơn vị:"}])
    boxes = {r["label"]: r for r in json.loads(sandbox["inv_boxes.json"])["boxes"]}
    assert boxes["Buyer"]["key_box"] == [20, 120, 105, 138] and boxes["Seller"]["key_box"] == [20, 20, 105, 38]


async def test_a_key_that_is_inside_the_value_box_is_not_boxed_again(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[{"label": "Dòng", "lines": [1], "key": "Mã số thuế:"}])   # no `text`: the whole line
    assert json.loads(sandbox["inv_boxes.json"])["boxes"][0]["key_box"] is None and "was not found" in out


async def test_boxes_cannot_be_drawn_on_an_image_that_has_not_been_read(sandbox):
    out = await call(tools.draw_boxes, path="inv.png", items=[{"label": "x", "lines": [1]}])
    assert out.startswith("Error: there is no OCR result") and "inv_boxes.png" not in sandbox


async def test_there_is_no_way_to_give_coordinates(sandbox):
    assert set(tools.BoxItem.model_fields) == {"label", "lines", "cells", "text", "key"}       # ids and text only: no coordinates
    assert "x0" not in json.dumps(tools.draw_boxes.args_schema.model_json_schema())


async def test_a_box_is_drawn_exactly_where_the_ocr_found_the_text(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[
        {"label": "Mã số thuế", "lines": [1], "text": "0309405347"},       # just the value
        {"label": "Tổng dòng", "lines": [2]}])                              # the whole line
    assert "Drew 2 boxes" in out and "inv_boxes.png" in sandbox
    assert "inv_boxes.png is shown to the user under this step" in out and "do not present_file" in out
    record = json.loads(sandbox["inv_boxes.json"])["boxes"]
    assert record[0]["box"] == [130, 40, 220, 58] and record[1]["box"] == [20, 100, 260, 118]
    img = Image.open(io.BytesIO(sandbox["inv_boxes.png"])).convert("RGB")
    assert img.getpixel((129, 49)) != (255, 255, 255)                      # the frame is on the value's left edge
    assert sandbox["inv.png"] == png((300, 200))                             # the original is untouched


async def test_what_could_not_be_placed_is_reported_not_dropped(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[
        {"label": "ok", "lines": [1]}, {"label": "ghost", "lines": [9]},
        {"label": "wrong text", "lines": [2], "text": "Mã số thuế"}])
    assert "Drew 1 box" in out and "Not drawn:" in out
    assert "ghost: no OCR line L9" in out and "wrong text: the text 'Mã số thuế' was not found" in out


async def test_an_image_changed_since_ocr_is_refused(sandbox):
    await call(tools.ocr_image, path="inv.png")
    sandbox["inv.png"] = png((300, 200), color="gray")
    assert "has changed since it was read by OCR" in await call(tools.draw_boxes, path="inv.png", items=[{"label": "a", "lines": [1]}])


async def test_paths_and_sizes_are_checked(sandbox):
    await call(tools.ocr_image, path="inv.png")
    assert "must be a different file" in await call(tools.draw_boxes, path="inv.png", output="inv.png", items=[{"label": "a", "lines": [1]}])
    assert "inside your working directory" in await call(tools.draw_boxes, path="inv.png", output="../x.png", items=[{"label": "a", "lines": [1]}])
    assert "at most 60" in await call(tools.draw_boxes, path="inv.png", items=[{"label": str(i), "lines": [1]} for i in range(61)])
    assert "is not in the sandbox" in await call(tools.ocr_image, path="nope.png")


# ---- the real tesseract, on text drawn at a known place -------------------------------------------------

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@pytest.mark.skipif(not ocr.available() or not os.path.exists(_FONT), reason="needs tesseract and the DejaVu font")
def test_real_ocr_puts_the_box_on_text_drawn_at_a_known_position():
    img = Image.new("RGB", (900, 400), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(_FONT, 34)
    d.text((60, 60), "Invoice number 4821", fill="black", font=f)
    d.text((60, 160), "Total amount 1.320.000 USD", fill="black", font=f)
    d.text((60, 260), "Mã số thuế 0309405347", fill="black", font=f)
    out = io.BytesIO(); img.save(out, "PNG")
    size, lines, _ = ocr.run_ocr(out.getvalue())
    assert size == (900, 400)
    for text, expect_x, expect_y in (("4821", 60 + d.textlength("Invoice number ", font=f), 60),
                                     ("1.320.000", 60 + d.textlength("Total amount ", font=f), 160),
                                     ("0309405347", 60 + d.textlength("Mã số thuế ", font=f), 260)):
        hits = [ocr.locate(l, text) for l in lines]
        box = max((h for h in hits if h), key=lambda h: h[2])[0]
        assert abs(box[0] - expect_x) <= 10 and abs(box[1] - expect_y) <= 20, (text, box, expect_x, expect_y)


# ---- bilingual glosses and number tags ---------------------------------------------------------------

def test_a_date_with_bracketed_translations_inside_is_boxed_as_a_whole():
    words = "Ngày (date) 02 tháng (month) 04 năm (year) 2026".split()
    tsv = "\n".join([HEADER] + [row(1, 1, 1, i, 10 + i * 60, 5, 50, 12, 90, w) for i, w in enumerate(words)])
    line = ocr.parse_tsv(tsv, 1)[0]
    box, matched, score = ocr.locate(line, "02 tháng 04 năm 2026")
    assert matched == "02 tháng (month) 04 năm (year) 2026" and score >= ocr.MATCH_THRESHOLD
    assert box[0] == line.words[2].box[0] and box[2] == line.words[-1].box[2]      # from "02" to "2026", glosses between


def _line(text: str, conf: float = 90) -> "ocr.Line":
    tsv = "\n".join([HEADER] + [row(1, 1, 1, i, 10 + i * 60, 5, 50, 12, conf, w) for i, w in enumerate(text.split())])
    return ocr.parse_tsv(tsv, 1)[0]


def test_a_number_with_two_digits_swapped_is_not_boxed_as_the_wanted_one():
    line = _line("Total 1.230.000 USD")
    assert ocr.locate(line, "1.320.000") is None
    assert ocr.locate(_line("Total 1.320.000 USD"), "1.320.000") is not None


def test_one_digit_more_or_less_is_not_a_match():
    assert ocr.locate(_line("No 40123"), "4012") is None
    assert ocr.locate(_line("No 401"), "4012") is None


def test_a_date_is_found_when_the_picture_writes_it_out_in_words():
    box, matched, _ = ocr.locate(_line("Date 02 April 04 2026 Paid"), "02/04/2026") or (None, "", 0)
    assert box is None or "2026" in matched          # digits 02,04,2026 in order are not all there: must not be forced
    hit = ocr.locate(_line("Ngay 02 thang 04 nam 2026 xong"), "02/04/2026")
    assert hit and hit[1].startswith("02") and hit[1].endswith("2026")


def test_a_digit_misread_as_a_look_alike_letter_is_repaired_but_words_are_left_alone():
    assert ocr.locate(_line("Total 1O5O USD"), "1050") is not None          # O read for 0
    assert ocr.locate(_line("Ref INV-4714 ok"), "INV-4714") is not None      # not turned into 1NV-4714


@pytest.mark.parametrize("text", ["請求書番号 2024", "Счёт номер 2024", "เลขที่ 2024"])
def test_text_in_other_scripts_is_matched_not_stripped_away(text):
    line = _line(text)
    hit = ocr.locate(line, text.split()[0])
    assert hit and hit[2] >= ocr.MATCH_THRESHOLD


def test_norm_keeps_every_script_and_only_strips_accents_from_latin():
    assert ocr.norm("Mã SỐ Đồng") == "ma so dong"
    assert ocr.norm("請求書") == "請求書"
    assert ocr.norm("Счёт") == ocr.norm("счёт") and ocr.norm("Счёт")


def test_check_lang_refuses_what_is_not_installed(monkeypatch):
    monkeypatch.setattr(ocr, "installed_langs", lambda: ["eng", "vie"])
    assert ocr.check_lang("eng+vie") == "eng+vie"
    with pytest.raises(ocr.OcrError):
        ocr.check_lang("klingon")


def test_a_number_tag_is_put_where_it_covers_no_text():
    text_above = (50, 60, 150, 78)                      # a line of text right above the box
    box = (50, 80, 150, 98)
    img = Image.open(io.BytesIO(ocr.draw_boxes(png((300, 200)), [("x", box, None)], obstacles=[text_above, box]))).convert("RGB")
    # above-left would cover the text above; the tag must not sit on that line's pixels
    assert all(img.getpixel((x, y)) == (255, 255, 255) for x in range(60, 150, 7) for y in (62, 70))
    spot = ocr._tag_spot((49, 79, 151, 99), (14, 14), [text_above, box], (300, 200))
    assert not ocr._overlap((spot[0], spot[1], spot[0] + 14, spot[1] + 14), text_above)



# ---- tables: found from their rules, read cell by cell, named T1.R2.C3 -------------------------------------

def grid_image(spans=None) -> Image.Image:
    """A 3x3 ruled table (x at 40/170/300/430, y at 40/100/160/220) and, under it, a wide ruled strip."""
    img = Image.new("RGB", (500, 330), "white")
    d = ImageDraw.Draw(img)
    for x in (40, 170, 300, 430):
        d.line((x, 40, x, 220), fill="black", width=2)
    for y in (40, 100, 160, 220):
        d.line((40, y, 430, y), fill="black", width=2)
    d.rectangle((40, 222, 430, 262), outline="black", width=2)         # a totals-style strip right under the table's last rule
    return img


def test_a_ruled_table_is_found_as_a_grid_of_cells_and_a_strip_under_it_is_not_part_of_it():
    from app.agents import tables as tb

    grids = tb.find_grids(grid_image())
    assert len(grids) == 1 and len(grids[0].cells) == 9
    assert grids[0].box[0] in range(38, 46) and grids[0].box[3] in range(214, 224)      # ends at the table's own last rule
    first = grids[0].cells[0]
    assert 38 <= first[0] <= 46 and 38 <= first[1] <= 46 and 165 <= first[2] <= 172 and 95 <= first[3] <= 102


def test_page_text_not_in_a_table_stays_as_lines_and_cell_text_goes_to_its_cell():
    from app.agents import tables as tb

    grid = tb.find_grids(grid_image())[0]
    keyed = [((1, 1, 1), ocr.Word("Hóa", (50, 5, 90, 25))),                    # above the table
             ((2, 1, 1), ocr.Word("A", (60, 60, 80, 80))),                     # inside cell R1.C1
             ((2, 1, 2), ocr.Word("B", (200, 60, 220, 80))),                   # inside cell R1.C2
             ((3, 1, 1), ocr.Word("C", (60, 180, 80, 200)))]                   # inside R3.C1
    tables, outside = ocr.build_tables([grid], keyed)
    assert [w.text for _, w in outside] == ["Hóa"]
    t = tables[0]
    assert (t.rows, t.cols) == (3, 3)
    texts = {(c.r, c.c): c.text for c in t.cells if c.text}
    assert texts == {(1, 1): "A", (1, 2): "B", (3, 1): "C"}


def test_a_table_with_no_text_is_dropped_and_a_merged_cell_gets_its_span():
    from app.agents import tables as tb

    grid = tb.Grid((40, 40, 430, 220), [(42, 42, 298, 98), (302, 42, 428, 98), (42, 102, 168, 158), (172, 102, 298, 158), (302, 102, 428, 158),
                                        (42, 162, 168, 218), (172, 162, 298, 218), (302, 162, 428, 218)])
    t, _ = ocr.build_tables([grid], [((1, 1, 1), ocr.Word("X", (60, 60, 80, 80)))])
    merged = next(c for c in t[0].cells if c.r == 1 and c.c == 1)
    assert merged.colspan == 2 and (t[0].rows, t[0].cols) == (3, 3)
    assert ocr.build_tables([grid], [])[0] == []


def test_references_name_a_cell_a_row_a_column_or_the_table():
    cells = lambda ref: ocr.resolve_cells(TABLES, ref)
    assert [c.text for c in cells("T1.R2.C3")] == ["1.320.000"]
    assert [c.text for c in cells("t1.r2")] == ["1", "Phần mềm", "1.320.000"]
    assert [c.text for c in cells("T1.C2")] == ["Tên hàng", "Phần mềm"]
    assert len(cells("T1")) == 6
    assert cells("T2") is None and cells("T1.R9") is None and cells("A1") is None


async def test_ocr_lists_the_table_separately_by_row_and_saves_it_as_csv(sandbox):
    out = await call(tools.ocr_image, path="inv.png")
    assert "2 lines of text outside tables and 1 table" in out and "Table T1 [10,130,290,190], 2 rows x 3 columns" in out
    assert "R2: C1 '1' | C2 'Phần mềm' | C3 '1.320.000'" in out
    csv_text = sandbox["inv.png.T1.csv"].decode("utf-8-sig")
    assert csv_text.splitlines() == ["STT,Tên hàng,Thành tiền", "1,Phần mềm,1.320.000"]
    assert json.loads(sandbox["inv.png.ocr.json"])["tables"][0]["cells"][5]["text"] == "1.320.000"


async def test_boxes_can_be_drawn_on_a_cell_a_row_or_the_whole_table(sandbox):
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[
        {"label": "Amount", "cells": ["T1.R2.C3"]}, {"label": "Item row", "cells": ["T1.R2"]},
        {"label": "Table", "cells": ["T1"]}, {"label": "Value only", "cells": ["T1.R2.C3"], "text": "1.320.000"},
        {"label": "Bad", "cells": ["T9.R1"]}])
    assert "Drew 4 boxes" in out and "Bad: no OCR line or cell T9.R1" in out
    boxes = {r["label"]: r["box"] for r in json.loads(sandbox["inv_boxes.json"])["boxes"]}
    assert boxes["Table"] == [10, 130, 290, 190]                                  # the table's own outline
    assert boxes["Amount"] == [204, 164, 240, 186]                                # the text inside the cell
    assert boxes["Item row"][0] == 14 and boxes["Item row"][2] == 240            # from the first cell's text to the last cell's
    assert boxes["Value only"] == boxes["Amount"]


def test_the_columns_come_from_the_fullest_row_not_from_an_oddly_placed_cell():
    from app.agents import tables as tb

    regular = [(42, 42, 168, 98), (172, 42, 298, 98), (302, 42, 428, 98), (42, 102, 168, 158), (172, 102, 298, 158), (302, 102, 428, 158)]
    odd = (250, 162, 428, 218)                                          # starts at x=250, in no column
    grid = tb.Grid((40, 40, 430, 220), regular + [odd])
    t, _ = ocr.build_tables([grid], [((1, 1, 1), ocr.Word("X", (60, 60, 80, 80)))])
    assert t[0].cols == 3                                              # not 4: the odd cell does not make a column


def test_the_tools_tell_the_model_to_label_with_the_field_name_and_to_pass_the_value_as_text():
    schema = json.dumps(tools.draw_boxes.args_schema.model_json_schema())
    assert "NAME of the field" in schema and "Not its value" in schema and "VALUE here" in schema
    assert "POSITIONS" in tools.ocr_image.description and "view_image" in tools.ocr_image.description


# ---- boxes add up across calls; an empty cell can still be boxed ------------------------------------------

async def test_a_second_call_adds_to_the_boxes_already_drawn_instead_of_replacing_them(sandbox):
    await call(tools.ocr_image, path="inv.png")
    await call(tools.draw_boxes, path="inv.png", items=[{"label": "A", "lines": [1]}, {"label": "B", "lines": [2]}])
    out = await call(tools.draw_boxes, path="inv.png", items=[{"label": "C", "cells": ["T1.R2.C3"]}])
    assert "added to the 2 already there (3 in all)" in out and "- C:" in out
    record = json.loads(sandbox["inv_boxes.json"])["boxes"]
    assert [r["label"] for r in record] == ["A", "B", "C"]
    img = Image.open(io.BytesIO(sandbox["inv_boxes.png"])).convert("RGB")
    for x, y in ((19, 49), (19, 109), (203, 175)):                          # a frame pixel of each of the three boxes
        assert img.getpixel((x, y)) != (255, 255, 255), (x, y)


async def test_a_box_with_the_same_label_replaces_the_earlier_one_and_replace_starts_over(sandbox):
    await call(tools.ocr_image, path="inv.png")
    await call(tools.draw_boxes, path="inv.png", items=[{"label": "A", "lines": [1]}, {"label": "B", "lines": [2]}])
    await call(tools.draw_boxes, path="inv.png", items=[{"label": "A", "lines": [2]}])
    record = json.loads(sandbox["inv_boxes.json"])["boxes"]
    assert [r["label"] for r in record] == ["B", "A"] and record[1]["box"] == [20, 100, 260, 118]
    await call(tools.draw_boxes, path="inv.png", items=[{"label": "Only", "lines": [1]}], replace=True)
    assert [r["label"] for r in json.loads(sandbox["inv_boxes.json"])["boxes"]] == ["Only"]


async def test_a_cell_named_outright_that_ocr_read_nothing_in_is_boxed_as_a_cell(sandbox, monkeypatch):
    empty = [ocr.Table(1, (10, 130, 290, 190), 2, 3, [_cell(1, 1, (10, 130, 60, 160), "STT"), ocr.Cell(2, 1, (10, 160, 60, 190), [])])]
    monkeypatch.setattr(ocr, "run_ocr", lambda data, lang=None, psm=None: ((300, 200), LINES, empty))
    await call(tools.ocr_image, path="inv.png")
    out = await call(tools.draw_boxes, path="inv.png", items=[{"label": "Quantity", "cells": ["T1.R2.C1"], "text": "1"}])
    assert "Drew 1 box" in out and "OCR read no text in T1.R2.C1, so the cell itself is boxed" in out
    assert json.loads(sandbox["inv_boxes.json"])["boxes"][0]["box"] == [10, 160, 60, 190]
    wrong = await call(tools.draw_boxes, path="inv.png", items=[{"label": "X", "lines": [1], "text": "nothing like it"}])
    assert "Error: nothing could be drawn" in wrong                          # a line is still not boxed on a text that is not in it


# ---- a page of a PDF: read from a rendered image, marked on the image and on the PDF itself ----------------

def _pdf(pages=2) -> bytes:
    imgs = [Image.new("RGB", (600, 800), "white") for _ in range(pages)]
    out = io.BytesIO()
    imgs[0].save(out, "PDF", save_all=True, append_images=imgs[1:], resolution=100)
    return out.getvalue()


@pytest.fixture
def pdf_sandbox(sandbox):
    sandbox["doc.pdf"] = _pdf(2)
    return sandbox


async def test_ocr_reads_a_chosen_page_of_a_pdf_from_a_rendered_image(pdf_sandbox):
    out = await call(tools.ocr_image, path="doc.pdf", page=2)
    assert "doc.p2.png" in pdf_sandbox and "Page 2 of 2" in out and "use doc.p2.png" in out
    saved = json.loads(pdf_sandbox["doc.p2.png.ocr.json"])
    assert saved["source"]["pdf"] == "doc.pdf" and saved["source"]["page"] == 2
    assert saved["sha256"] == ocr.sha256(pdf_sandbox["doc.p2.png"])


async def test_a_pdf_page_out_of_range_and_page_on_an_image_are_refused(pdf_sandbox):
    assert "has 2 pages; page 5 does not exist" in await call(tools.ocr_image, path="doc.pdf", page=5)
    assert (await call(tools.ocr_image, path="inv.png", page=1)).startswith("Error: `page` is only for PDFs")


async def test_boxes_drawn_on_a_pdf_page_are_laid_over_the_pdf_and_pages_add_up(pdf_sandbox):
    import pypdfium2 as pdfium

    await call(tools.ocr_image, path="doc.pdf", page=2)
    out = await call(tools.draw_boxes, path="doc.p2.png", items=[{"label": "Thuế", "lines": [1], "text": "0309405347"}])
    assert "doc.p2_boxes.png" in pdf_sandbox and "doc_boxes.pdf" in pdf_sandbox
    assert "pages marked so far: 2" in out and "doc_boxes.pdf is shown to the user" in out

    marked = pdfium.PdfDocument(pdf_sandbox["doc_boxes.pdf"])
    assert len(marked) == 2                                                       # the PDF's own pages, no legend page added
    plain = pdfium.PdfDocument(pdf_sandbox["doc.pdf"])
    assert marked[1].render(scale=1).to_pil().tobytes() != plain[1].render(scale=1).to_pil().tobytes()   # page 2 carries the box
    assert marked[0].render(scale=1).to_pil().tobytes() == plain[0].render(scale=1).to_pil().tobytes()   # page 1 untouched

    await call(tools.ocr_image, path="doc.pdf", page=1)
    out = await call(tools.draw_boxes, path="doc.p1.png", items=[{"label": "Tổng", "lines": [2]}])
    assert "pages marked so far: 1, 2" in out
    marked = pdfium.PdfDocument(pdf_sandbox["doc_boxes.pdf"])
    assert len(marked) == 2
    assert marked[0].render(scale=1).to_pil().tobytes() != plain[0].render(scale=1).to_pil().tobytes()   # now page 1 too


async def test_a_pdf_that_changed_since_it_was_read_is_not_marked(pdf_sandbox):
    await call(tools.ocr_image, path="doc.pdf", page=1)
    pdf_sandbox["doc.pdf"] = _pdf(3)
    out = await call(tools.draw_boxes, path="doc.p1.png", items=[{"label": "x", "lines": [1]}])
    assert "has changed since this page was read" in out and "doc_boxes.pdf" not in pdf_sandbox
    assert "doc.p1_boxes.png" in pdf_sandbox                                           # the page image is still there


async def test_pages_marked_in_parallel_calls_all_end_up_in_the_pdf(pdf_sandbox):
    import asyncio
    import pypdfium2 as pdfium

    pdf_sandbox["doc.pdf"] = _pdf(4)
    for n in (1, 2, 3):
        await call(tools.ocr_image, path="doc.pdf", page=n)
    await asyncio.gather(*[call(tools.draw_boxes, path=f"doc.p{n}.png", items=[{"label": f"box {n}", "lines": [1]}]) for n in (1, 2, 3)])
    state = json.loads(pdf_sandbox["doc_boxes.json"])
    assert sorted(state["pages"]) == ["1", "2", "3"]
    plain, marked = pdfium.PdfDocument(pdf_sandbox["doc.pdf"]), pdfium.PdfDocument(pdf_sandbox["doc_boxes.pdf"])
    for i in (0, 1, 2):
        assert marked[i].render(scale=1).to_pil().tobytes() != plain[i].render(scale=1).to_pil().tobytes()


# ---- the PaddleOCR engine: words from its tokens, the language, tight boxes, the choice of engine ------------------

from app.agents import ocr_paddle as paddle


def _quad(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_paddle_tokens_are_put_back_into_words():
    tokens = ["K", "ý ", "hiu", " (", "Serial", "):"]
    regions = [_quad(0, 0, 8, 20), _quad(9, 0, 17, 20), _quad(30, 0, 50, 20), _quad(55, 0, 60, 20), _quad(60, 0, 100, 20), _quad(100, 0, 110, 20)]
    assert paddle._words(tokens, regions) == [("Ký", (0, 0, 17, 20)), ("hiu", (30, 0, 50, 20)), ("(Serial):", (55, 0, 110, 20))]


def test_paddle_cjk_comes_out_a_character_at_a_time_and_a_lone_space_ends_a_word():
    words = paddle._words(["請", "求", " ", "書"], [_quad(0, 0, 10, 10), _quad(10, 0, 20, 10), _quad(20, 0, 24, 10), _quad(24, 0, 34, 10)])
    assert [w for w, _ in words] == ["請求", "書"]


def test_paddle_language_is_the_first_one_that_is_not_english():
    assert paddle.paddle_lang("vie+eng") == "vi" and paddle.paddle_lang("eng+jpn") == "japan"
    assert paddle.paddle_lang("eng") == "en" and paddle.paddle_lang("") == "en" and paddle.paddle_lang("xxx") == "en"


def test_a_loose_box_is_shrunk_to_the_ink_and_an_empty_one_is_left_alone():
    import numpy as np

    gray = np.full((100, 200), 255, dtype=np.uint8)
    gray[40:52, 60:120] = 0                                          # the text
    assert paddle.tighten(gray, (50, 30, 140, 70), pad=0) == (60, 40, 120, 52)
    dark = np.zeros((100, 200), dtype=np.uint8)
    dark[40:52, 60:120] = 255                                        # light text on a dark ground counts too
    assert paddle.tighten(dark, (50, 30, 140, 70), pad=0) == (60, 40, 120, 52)
    assert paddle.tighten(gray, (0, 0, 30, 20)) == (0, 0, 30, 20)    # nothing there: unchanged


def test_the_box_of_a_word_on_a_tilted_line_follows_the_tilt():
    # a line tilted by ~3 degrees as PaddleOCR gives it (the figures are from a real photographed page):
    # its word comes back as a 6 px strip, but the letters span the line's edges at the word's own ends
    poly = [[478, 456], [955, 432], [957, 462], [480, 486]]
    assert paddle.follow_slant((707, 456, 946, 462), poly) == (707, 432, 946, 475)      # was 6 px high
    assert paddle.follow_slant((480, 450, 500, 470), poly)[1] <= 450                    # never smaller than it came
    assert paddle.follow_slant((100, 10, 200, 40), [[0, 10], [300, 10], [300, 40], [0, 40]]) == (100, 10, 200, 40)  # level: as it was
    assert paddle.follow_slant((10, 10, 40, 80), [[0, 0], [5, 90], [35, 90], [30, 0]]) == (10, 10, 40, 80)          # text running down the page: left alone
    assert paddle.follow_slant((1, 2, 3, 4), [[0, 0], [1, 1]]) == (1, 2, 3, 4)          # not a quadrilateral


def test_paddle_is_used_when_installed_and_tesseract_when_asked_for_or_missing(monkeypatch):
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    monkeypatch.setattr(paddle, "available", lambda: True)
    assert ocr.engine_name() == "paddle" and ocr.engine_name("tesseract") == "tesseract"
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    assert ocr.engine_name() == "tesseract"
    monkeypatch.delenv("OCR_ENGINE")
    monkeypatch.setattr(paddle, "available", lambda: False)
    assert ocr.engine_name() == "tesseract"


def test_run_ocr_reads_with_paddle_and_puts_its_words_in_lines(monkeypatch):
    monkeypatch.setattr(ocr, "engine_name", lambda engine=None: "paddle")
    monkeypatch.setattr(paddle, "read", lambda img, lang: [("Tổng cộng 1.320.000", 0.97, [("Tổng", (10, 10, 50, 28)), ("cộng", (56, 10, 96, 28)), ("1.320.000", (150, 10, 240, 28))]),
                                                          ("Mã số thuế", 0.9, [("Mã", (10, 60, 30, 78)), ("số", (36, 60, 56, 78)), ("thuế", (62, 60, 100, 78))])])
    size, lines, tables = ocr.run_ocr(png((300, 200)), "vie+eng")
    assert size == (300, 200) and tables == [] and [l.text for l in lines] == ["Tổng cộng 1.320.000", "Mã số thuế"]
    box, matched, _ = ocr.locate(lines[0], "1.320.000")
    assert box == (150, 10, 240, 28)


def test_an_unsupported_language_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(ocr, "engine_name", lambda engine=None: "paddle")
    with pytest.raises(ocr.OcrError, match="klingon"):
        ocr.run_ocr(png((100, 100)), "klingon")


def test_paddle_failing_on_a_picture_falls_back_to_tesseract(monkeypatch):
    monkeypatch.setattr(ocr, "engine_name", lambda engine=None: "paddle")
    monkeypatch.setattr(ocr, "_run_paddle", lambda data, lang: (_ for _ in ()).throw(RuntimeError("model would not load")))
    monkeypatch.setattr(ocr, "_run_tesseract", lambda data, lang, psm, scale: ((9, 9), [], []))
    monkeypatch.setattr(ocr, "available", lambda: True)
    assert ocr.run_ocr(png((10, 10)))[0] == (9, 9)
