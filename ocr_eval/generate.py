"""Synthetic documents with known text positions, for measuring how well boxes land.

Each case is an image plus targets: a piece of text as it is printed and the true box it was drawn in
(a target with box None is a DECOY, text that is NOT in the image, used to check that a near-miss is
not boxed). Content is random but seeded, so a case is the same every run. Seeds 1-3 are for tuning
settings and 101-103 are held out: look at those only to report, never to tune on.

The categories differ on purpose from the one invoice the OCR tools were first tuned on: another
language and script, no ruled table, a receipt, a statement full of similar numbers, a rotated scan, a
low-resolution noisy one.
"""

import io
import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "thai": "/usr/share/fonts/opentype/tlwg/Loma.otf",
}
Box = tuple[float, float, float, float]


@dataclass
class Target:
    label: str
    text: str
    box: Box | None


@dataclass
class Case:
    name: str
    category: str
    lang: str                      # the OCR language that suits this document
    image: Image.Image
    targets: list[Target]


def font(kind: str, size: int):
    return ImageFont.truetype(FONT[kind], size)


class Page:
    def __init__(self, w: int, h: int):
        self.img = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s: str, f, fill="black") -> Box:
        self.d.text(xy, s, font=f, fill=fill)
        return tuple(float(v) for v in self.d.textbbox(xy, s, font=f))

    def after(self, xy, label: str, value: str, f) -> Box:
        """'label value' on one line; returns the box of the value only."""
        self.d.text(xy, label, font=f, fill="black")
        x = xy[0] + self.d.textlength(label + " ", font=f)
        return self.text((x, xy[1]), value, f)


def swap_digits(s: str) -> str:
    """The same text with two different neighbouring digits swapped: a plausible OCR-or-typo near miss."""
    chars = list(s)
    idx = [i for i in range(len(chars) - 1) if chars[i].isdigit() and chars[i + 1].isdigit() and chars[i] != chars[i + 1]]
    if not idx:
        return s + "0"
    i = idx[len(idx) // 2]
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def money(rng, lo, hi, dec=2, sep=",", dot="."):
    v = rng.randint(lo * 100, hi * 100) / 100 if dec else rng.randint(lo, hi)
    s = f"{v:,.{dec}f}"
    return s.replace(",", "\x00").replace(".", dot).replace("\x00", sep)


COMPANIES = ["Northwind Traders", "Blue Harbor Supplies", "Maple Ridge Consulting", "Alder & Finch Ltd", "Summit Peak Logistics", "Cedar Point Foods"]
ITEMS = ["Consulting services", "Office chairs", "Printer toner", "Network switch", "Cloud hosting", "Training session", "Safety gloves"]
VI_COMPANIES = ["CÔNG TY TNHH THƯƠNG MẠI ĐẠI PHÁT", "CÔNG TY CỔ PHẦN GIẢI PHÁP SỐ VIỆT", "CÔNG TY TNHH SẢN XUẤT HOÀNG GIA", "DOANH NGHIỆP TƯ NHÂN MINH ANH"]
VI_ADDR = ["Số 12, đường Nguyễn Huệ, Quận 1, TP Hồ Chí Minh", "45 Lê Lợi, Phường Bến Thành, Quận 1, TP Hồ Chí Minh", "Tầng 5, 128 Hoàng Quốc Việt, Cầu Giấy, Hà Nội"]
VI_ITEMS = ["Phần mềm kế toán", "Dịch vụ tư vấn thuế", "Máy in laser", "Bảo trì hệ thống", "Đào tạo nhân sự"]


def ruled_table(page: Page, x, y, widths, rows, f, header=None, row_h=44):
    """Draws a grid and the text in it; returns {(r, c): text box}."""
    d = page.d
    xs = [x]
    for w in widths:
        xs.append(xs[-1] + w)
    total_rows = len(rows) + (1 if header else 0)
    for r in range(total_rows + 1):
        d.line((xs[0], y + r * row_h, xs[-1], y + r * row_h), fill="black", width=2)
    for xx in xs:
        d.line((xx, y, xx, y + total_rows * row_h), fill="black", width=2)
    boxes = {}
    allrows = ([header] if header else []) + rows
    for r, row in enumerate(allrows):
        for c, cell in enumerate(row):
            right = c >= 1 and any(ch.isdigit() for ch in cell) and not header_like(cell)
            tw = d.textlength(cell, font=f)
            tx = xs[c + 1] - tw - 12 if right else xs[c] + 12
            boxes[(r, c)] = page.text((tx, y + r * row_h + 10), cell, f)
    return boxes


def header_like(s):
    return not any(ch.isdigit() for ch in s)


def invoice_en(seed: int) -> Case:
    rng = random.Random(seed)
    p = Page(820, 900)
    f, big = font("sans", 20), font("sans", 36)
    p.text((40, 30), "INVOICE", big)
    no, date = f"INV-{rng.randint(1000, 9999)}", f"{rng.choice(['March', 'June', 'October', 'January'])} {rng.randint(1, 28)}, 2026"
    company = rng.choice(COMPANIES)
    t_no = p.after((40, 100), "Invoice No:", no, f)
    t_date = p.after((40, 135), "Date:", date, f)
    t_co = p.after((40, 170), "Bill to:", company, f)
    rows = [[rng.choice(ITEMS), str(rng.randint(1, 20)), money(rng, 10, 900), money(rng, 100, 9000)] for _ in range(4)]
    cells = ruled_table(p, 40, 250, [340, 90, 150, 160], rows, f, header=["Description", "Qty", "Unit price", "Amount"])
    total = money(rng, 1000, 20000)
    t_total = p.after((560, 250 + 5 * 44 + 30), "Total:", total, f)
    targets = [Target("invoice no", no, t_no), Target("date", date, t_date), Target("company", company, t_co),
               Target("amount r2", rows[1][3], cells[(2, 3)]), Target("amount r4", rows[3][3], cells[(4, 3)]), Target("total", total, t_total),
               Target("decoy total", swap_digits(total), None), Target("decoy invoice no", swap_digits(no), None)]
    return Case(f"invoice_en-{seed}", "invoice_en", "eng", p.img, targets)


def invoice_vi(seed: int) -> Case:
    rng = random.Random(seed)
    p = Page(900, 980)
    f, big = font("serif", 22), font("serif", 34)
    p.text((260, 30), "HÓA ĐƠN BÁN HÀNG", big)
    day, month = rng.randint(1, 28), rng.randint(1, 12)
    p.d.text((40, 100), "Ngày", font=f, fill="black")
    t_date = p.text((40 + p.d.textlength("Ngày ", font=f), 100), f"{day:02d} tháng {month:02d} năm 2026", f)
    seller = rng.choice(VI_COMPANIES)
    mst = f"0{rng.randint(300000000, 399999999)}"
    addr = rng.choice(VI_ADDR)
    t_seller = p.after((40, 140), "Đơn vị bán hàng (Issued):", seller, f)
    t_mst = p.after((40, 180), "Mã số thuế (Tax code):", mst, f)
    t_addr = p.after((40, 220), "Địa chỉ (Address):", addr, f)
    rows = [[str(i + 1), rng.choice(VI_ITEMS), str(rng.randint(1, 9)), money(rng, 100, 900, 0, ".", ","), money(rng, 1000, 9000, 0, ".", ",")] for i in range(4)]
    cells = ruled_table(p, 40, 290, [60, 330, 100, 170, 170], rows, f, header=["STT", "Tên hàng hóa", "SL", "Đơn giá", "Thành tiền"])
    total = money(rng, 5000, 30000, 0, ".", ",")
    t_total = p.after((520, 290 + 5 * 44 + 36), "Tổng cộng:", total, f)
    targets = [Target("date as digits", f"{day:02d}/{month:02d}/2026", t_date), Target("seller", seller, t_seller), Target("tax code", mst, t_mst),
               Target("address", addr, t_addr), Target("amount r1", rows[0][4], cells[(1, 4)]), Target("total", total, t_total),
               Target("decoy tax code", swap_digits(mst), None), Target("decoy total", swap_digits(total), None)]
    return Case(f"invoice_vi-{seed}", "invoice_vi", "vie+eng", p.img, targets)


def receipt(seed: int) -> Case:
    rng = random.Random(seed)
    p = Page(430, 760)
    f, big = font("mono", 19), font("mono", 24)
    p.text((120, 24), rng.choice(["CORNER MARKET", "GREEN LEAF CAFE", "CITY HARDWARE"]), big)
    when = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}"
    t_when = p.text((30, 80), when, f)
    y, prices, boxes = 130, [], []
    for _ in range(6):
        name, price = rng.choice(ITEMS)[:16], money(rng, 1, 60)
        p.text((30, y), name, f)
        w = p.d.textlength(price, font=f)
        boxes.append(p.text((400 - w, y), price, f))
        prices.append(price)
        y += 34
    total = money(rng, 60, 300)
    p.text((30, y + 20), "TOTAL", big)
    w = p.d.textlength(total, font=big)
    t_total = p.text((400 - w, y + 20), total, big)
    last4 = f"{rng.randint(1000, 9999)}"
    p.d.text((30, y + 90), "CARD **** **** ****", font=f, fill="black")
    t_card = p.text((30 + p.d.textlength("CARD **** **** **** ", font=f), y + 90), last4, f)
    targets = [Target("time", when, t_when), Target("price 2", prices[1], boxes[1]), Target("price 5", prices[4], boxes[4]), Target("total", total, t_total),
               Target("card", last4, t_card), Target("decoy total", swap_digits(total), None)]
    return Case(f"receipt-{seed}", "receipt", "eng", p.img, targets)


def statement(seed: int) -> Case:
    rng = random.Random(seed)
    p = Page(940, 640)
    f = font("sans", 18)
    cols = [30, 150, 470, 620, 770]
    for x, h in zip(cols, ["Date", "Description", "Debit", "Credit", "Balance"]):
        p.text((x, 24), h, f)
    p.d.line((20, 56, 920, 56), fill="black", width=2)
    bal, rows = rng.randint(2000, 9000), []
    boxes = {}
    for i in range(12):
        debit, credit = (rng.randint(10, 400), 0) if rng.random() < 0.6 else (0, rng.randint(10, 900))
        bal = bal - debit + credit
        vals = [f"2026-03-{i + 1:02d}", rng.choice(ITEMS), f"{debit:,}.00" if debit else "", f"{credit:,}.00" if credit else "", f"{bal:,}.00"]
        for c, v in enumerate(vals):
            if v:
                b = p.text((cols[c], 70 + i * 40), v, f)
                boxes[(i, c)] = b
        rows.append(vals)
    targets = [Target("balance r3", rows[2][4], boxes[(2, 4)]), Target("balance r7", rows[6][4], boxes[(6, 4)]), Target("balance r11", rows[10][4], boxes[(10, 4)]),
               Target("date r5", rows[4][0], boxes[(4, 0)]), Target("decoy balance", swap_digits(rows[6][4]), None), Target("decoy off-by-one", rows[2][4].replace(rows[2][4][-4], str((int(rows[2][4][-4]) + 1) % 10), 1) if rows[2][4][-4].isdigit() else rows[2][4] + "1", None)]
    return Case(f"statement-{seed}", "statement", "eng", p.img, targets)


def _rotate(case: Case, degrees: float, name: str) -> Case:
    img = case.image
    w, h = img.size
    out = img.rotate(degrees, expand=True, fillcolor="white", resample=Image.BICUBIC)
    ow, oh = out.size
    a = math.radians(degrees)

    def move(x, y):                                  # PIL rotates counter-clockwise about the centre
        dx, dy = x - w / 2, y - h / 2
        return (ow / 2 + dx * math.cos(a) + dy * math.sin(a), oh / 2 - dx * math.sin(a) + dy * math.cos(a))

    targets = []
    for t in case.targets:
        if t.box is None:
            targets.append(t)
            continue
        pts = [move(t.box[0], t.box[1]), move(t.box[2], t.box[1]), move(t.box[0], t.box[3]), move(t.box[2], t.box[3])]
        targets.append(Target(t.label, t.text, (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))))
    return Case(name, case.category, case.lang, out, targets)


def skew(seed: int) -> Case:
    base = invoice_en(seed + 7)
    angle = random.Random(seed).choice([-2.5, -1.5, 1.5, 2.5])
    c = _rotate(base, angle, f"skew-{seed}")
    c.category = "skew"
    return c


def noisy(seed: int) -> Case:
    base = invoice_vi(seed + 11)
    rng = random.Random(seed)
    s = 0.6
    small = base.image.resize((int(base.image.width * s), int(base.image.height * s)), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    px = small.load()
    for _ in range(small.width * small.height // 40):
        x, y = rng.randrange(small.width), rng.randrange(small.height)
        v = rng.randint(120, 255)
        px[x, y] = (v, v, v)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=35)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    targets = [Target(t.label, t.text, None if t.box is None else tuple(v * s for v in t.box)) for t in base.targets]
    return Case(f"noisy-{seed}", "noisy", "vie+eng", img, targets)


def _form(seed: int, name: str, lang: str, kind: str, size: int, rows: list[tuple[str, str]]) -> Case:
    p = Page(900, 120 + len(rows) * 80)
    f = font(kind, size)
    targets = []
    for i, (label, value) in enumerate(rows):
        targets.append(Target(label, value, p.after((40, 50 + i * 80), label, value, f)))
    swapped = swap_digits(rows[1][1])
    if swapped != rows[1][1]:
        targets.append(Target("decoy", swapped, None))
    return Case(f"{name}-{seed}", name, lang, p.img, targets)


def forms_ja(seed: int) -> Case:
    r = random.Random(seed)
    return _form(seed, "forms_ja", "jpn", "cjk", 30, [("請求書番号:", f"2026-{r.randint(1000, 9999)}"), ("合計金額:", f"¥{r.randint(10, 99)},{r.randint(100, 999)}"),
                                                       ("会社名:", r.choice(["株式会社サンプル商事", "東京テクノ有限会社", "大阪物流株式会社"]))])


def forms_zh(seed: int) -> Case:
    r = random.Random(seed)
    return _form(seed, "forms_zh", "chi_sim", "cjk", 30, [("发票号码:", f"2026{r.randint(10000, 99999)}"), ("合计:", f"¥{r.randint(1, 99)},{r.randint(100, 999)}.00"),
                                                           ("公司:", r.choice(["北京示例科技有限公司", "上海远航贸易有限公司", "深圳明辉电子有限公司"]))])


def forms_ru(seed: int) -> Case:
    r = random.Random(seed)
    return _form(seed, "forms_ru", "rus", "sans", 28, [("Счёт №:", f"2026-{r.randint(1000, 9999)}"), ("Итого:", f"{r.randint(10, 99)} {r.randint(100, 999)},00 руб."),
                                                       ("Компания:", r.choice(["ООО Северный путь", "АО Волга Трейд", "ООО Новый горизонт"]))])


def forms_th(seed: int) -> Case:
    r = random.Random(seed)
    return _form(seed, "forms_th", "tha", "thai", 30, [("เลขที่ใบแจ้งหนี้:", f"2026-{r.randint(1000, 9999)}"), ("ยอดรวม:", f"{r.randint(10, 99)},{r.randint(100, 999)} บาท"),
                                                     ("บริษัท:", r.choice(["บริษัท ตัวอย่าง จำกัด", "บริษัท สยามเทรด จำกัด", "ห้างหุ้นส่วน ไทยพัฒนา"]))])


CATEGORIES = {"invoice_en": invoice_en, "invoice_vi": invoice_vi, "receipt": receipt, "statement": statement, "skew": skew, "noisy": noisy,
              "forms_ja": forms_ja, "forms_zh": forms_zh, "forms_ru": forms_ru, "forms_th": forms_th}
TUNE_SEEDS = (1, 2, 3)
HELD_OUT_SEEDS = (101, 102, 103)


def cases(split: str) -> list[Case]:
    seeds = {"tune": TUNE_SEEDS, "heldout": HELD_OUT_SEEDS, "both": TUNE_SEEDS + HELD_OUT_SEEDS}[split]
    return [make(s) for make in CATEGORIES.values() for s in seeds]
