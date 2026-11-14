---
name: office-documents
description: Read, check or recalculate a Word, Excel or PowerPoint file in the bash sandbox. Use when the user gives a .docx, .xlsx, .pptx (or .doc, .xls, .ppt, .odt, .ods, .odp) and you need its text, its cells, how it looks, or computed formula values.
---
# Word, Excel and PowerPoint files

To read these files, three tools do the work on the server (they also recalculate formulas and draw pages): `view_image` is always available, `office_convert` and `office_recalculate` are loaded on demand, so first call `tool_search` with `select:office_convert,office_recalculate`. To make or edit a file yourself, python-docx, openpyxl, python-pptx and reportlab are installed in the sandbox.

1. If the file is in the user's workspace, `workspace_copy_to_sandbox` copies it into the sandbox.
2. To get the content as plain files you can read with bash:
   - a document: `office_convert(path, "txt")`, then `grep`/`sed -n` on the .txt
   - a workbook: `office_convert(path, "csv")` writes one csv per sheet (`book-Sheet1.csv`...); use `sheet=N` for one sheet; then read with pandas
   - a presentation: `office_convert(path, "pdf")`, then `check_pdf` and `read_pdf` on the result
3. To see how it looks (layout, charts, formatting, a slide): `view_image(path, pages=[N])` draws pages or slides as images, at most 5 per call. It needs a vision-capable model.
4. A workbook you wrote with a script has formulas but no computed values until it is opened by a spreadsheet program. `office_recalculate(path)` opens it, recalculates and saves `<name>-recalculated.xlsx`; then `office_convert` that file to csv to read the numbers. Never quote a value from a script-written workbook without doing this.
5. To hand the user a converted file, use `present_file`.

Say which pages, sheets or slides your answer comes from. If a conversion fails the file may be damaged or password-protected; tell the user rather than guessing at its content.
