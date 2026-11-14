---
name: spreadsheet-analysis
description: Answer a question about a spreadsheet or CSV in the user's file workspace by fetching it and computing with pandas. Use for totals, filters, group-bys, comparisons across sheets, or anything where the numbers matter.
---
# Spreadsheet analysis

A workbook has no extracted text, so `workspace_read_file` and `workspace_search_files` will not show its cells. Fetch it and compute.

1. `workspace_list_files` or `workspace_search_files` to find it, then `workspace_copy_to_sandbox` to copy it into your bash working directory.
2. Look before computing: sheet names, the header row, column dtypes, row count. A merged title row or a notes block above the table is common; find where the real header is before trusting `read_excel` defaults.
   ```python
   import pandas as pd
   x = pd.ExcelFile("data.xlsx"); print(x.sheet_names)
   df = x.parse(x.sheet_names[0], header=None); print(df.head(15))
   ```
3. Compute with code, never by eye. Print the numbers you will quote.
4. Check the result against the sheet: row counts after a filter, a total against the sheet's own total row if it has one. Blank cells, text stored as numbers and duplicate rows are the usual causes of a wrong answer.
5. Answer with the figure, how it was derived (which sheet, which filter), and anything in the data that made you choose an interpretation.

If the user wants the result as a file, write it with pandas (`to_excel` / `to_csv`) and hand it over with `present_file`.
