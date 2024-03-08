# ---------------------------------------------------------------------------------------------------------------
# GOOGLE SHEETS WRITE -------------------------------------------------------------------------------------------

import gspread

gc = gspread.service_account(filename="Credentials/google-sheets-api.json")

sh = gc.open("Salary_2023-Shiny-Python")
worksheet = sh.sheet1

headers = [
    "Salary",
    "Country",
    "EdLevel",
    "YearsCodePro",
    "DevType",
    "OrgSize",
    "OpSys",
    "Age",
    "APL",
    "Ada",
    "Apex",
    "Assembly",
    "Bash/Shell (all shells)",
    "C",
    "C#",
    "C++",
    "Clojure",
    "Cobol",
    "Crystal",
    "Dart",
    "Delphi",
    "Elixir",
    "Erlang",
    "F#",
    "Flow",
    "Fortran",
    "GDScript",
    "Go",
    "Groovy",
    "HTML/CSS",
    "Haskell",
    "Java",
    "JavaScript",
    "Julia",
    "Kotlin",
    "Lisp",
    "Lua",
    "MATLAB",
    "Nim",
    "OCaml",
    "Objective-C",
    "PHP",
    "Perl",
    "PowerShell",
    "Prolog",
    "Python",
    "R",
    "Raku",
    "Ruby",
    "Rust",
    "SAS",
    "SQL",
    "Scala",
    "Solidity",
    "Swift",
    "TypeScript",
    "VBA",
    "Visual Basic (.Net)",
    "Zig",
]

# worksheet.update("A1", [headers]) # to create headers

user = [
    34000,
    "Slovakia",
    "Master’s degree",
    12.0,
    "Developer, desktop or enterprise applications",
    "Just me - I am a freelancer, sole proprietor, etc.",
    "Windows",
    "35-44",
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

worksheet.append_row(user)

# ---------------------------------------------------------------------------------------------------------------
# GOOGLE SHEETS READ ------------------------------------------------------------------------------------------

import gspread
import pandas as pd

gc = gspread.service_account(filename="Credentials/google-sheets-api.json")

sh = gc.open("Salary_2023-Shiny-Python")
worksheet = sh.sheet1

records = worksheet.get_all_records()

df = pd.DataFrame(records)
df
