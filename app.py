# APP 2023
# https://shinylive.io/py/examples/#shinyswatch

# INSTALLATION
# conda create -n salary python=3.11
# conda env remove --name salary

# SHINY
# extension "Shiny for Python"
# pip install shiny
# shiny run

# DEPLOYMENT
# https://login.shinyapps.io/login
# pip install rsconnect-python
# pip freeze > requirements.txt
# conda env export --name salary > environment.yml (optional)
# rsconnect add --account jaroslavkotrba --name jaroslavkotrba --token XXXXXXXXXXXXXXXXXXXXXXXXXX --secret XXXXXXXXXXXXXXXXXXXXXXXXXX
# rsconnect list (optional)
# rsconnect deploy shiny . --entrypoint app:app --name jaroslavkotrba --title "salary_2023"

# GOOGLE SHEETS
# https://docs.google.com/spreadsheets/d/1tw7vAcSoo2vysrsIEGk03xZVdVWYpqtRBcuWcFs3O4k/edit#gid=0
# https://www.youtube.com/watch?v=zCEJurLGFRk

# Service Accounts
# https://console.cloud.google.com/apis/credentials?authuser=1&project=shiny-salary-2023-in-python
# Enable APIs & services -> ENABLE APIS AND SERVICES -> search Google Sheets API and Google Drive API -> ENABLE -> CREATE CREDENTIALS (after enabling) -> Application data NEXT -> Salary 2023 Shiny in Python -> CREATE AND CONTINUE -> Editor -> CONTINUE -> DONE

# Get Keys
# Enable APIs & services -> click on Service account -> KEYS -> Json -> CREATE -> put .json in folder Credentials

# Add email to the folder (keep restricted)
# e.g. salary-2023-shiny-in-python@shiny-salary-2023-in-python.iam.gserviceaccount.com

# Libraries
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib gspread

# TASKS
# TODO: new UI structure

# Import
import shinyswatch
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from faicons import icon_svg  # play sign icon

from shinywidgets import output_widget, register_widget, render_widget  # plots
import asyncio

app_ui = ui.page_navbar(
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate,
    #  solar, spacelab, superhero, united, vapor, yeti, zephyr
    shinyswatch.theme.sketchy(),
    ui.head_content(
        ui.tags.title("Salary of CODERs"),
        ui.tags.link(
            rel="icon",
            type="image/jpeg",
            href="https://freeiconshop.com/wp-content/uploads/edd/wallet-outline-filled.png",
        ),
    ),
    ui.nav_panel(
        "- PREDICTION -",
        ui.layout_sidebar(
            ui.sidebar(
                ui.tags.h2("Inputs for the model:"),
                ui.input_select(
                    id="country",
                    label="Select Country",
                    selected="Czech Republic",
                    choices={
                        "Argentina": "Argentina",
                        "Australia": "Australia",
                        "Austria": "Austria",
                        "Bangladesh": "Bangladesh",
                        "Belgium": "Belgium",
                        "Brazil": "Brazil",
                        "Bulgaria": "Bulgaria",
                        "Canada": "Canada",
                        "Chile": "Chile",
                        "China": "China",
                        "Colombia": "Colombia",
                        "Croatia": "Croatia",
                        "Czech Republic": "Czech Republic",
                        "Denmark": "Denmark",
                        "Estonia": "Estonia",
                        "Finland": "Finland",
                        "France": "France",
                        "Germany": "Germany",
                        "Greece": "Greece",
                        "Hong Kong (S.A.R.)": "Hong Kong (S.A.R.)",
                        "Hungary": "Hungary",
                        "India": "India",
                        "Iran": "Iran",
                        "Ireland": "Ireland",
                        "Israel": "Israel",
                        "Italy": "Italy",
                        "Japan": "Japan",
                        "Lithuania": "Lithuania",
                        "Malaysia": "Malaysia",
                        "Mexico": "Mexico",
                        "Netherlands": "Netherlands",
                        "New Zealand": "New Zealand",
                        "Norway": "Norway",
                        "Pakistan": "Pakistan",
                        "Poland": "Poland",
                        "Portugal": "Portugal",
                        "Romania": "Romania",
                        "Russia": "Russia",
                        "Serbia": "Serbia",
                        "Singapore": "Singapore",
                        "Slovakia": "Slovakia",
                        "Slovenia": "Slovenia",
                        "South Africa": "South Africa",
                        "South Korea": "South Korea",
                        "Spain": "Spain",
                        "Sweden": "Sweden",
                        "Switzerland": "Switzerland",
                        "Taiwan": "Taiwan",
                        "Turkey": "Turkey",
                        "UK": "UK",
                        "USA": "USA",
                        "Ukraine": "Ukraine",
                        "United Arab Emirates": "United Arab Emirates",
                    },
                ),
                ui.input_select(
                    "education",
                    "Select Education",
                    selected="Master’s degree",
                    choices={
                        "Less than a Bachelor": "Less than a Bachelor",
                        "Bachelor’s degree": "Bachelor’s degree",
                        "Master’s degree": "Master’s degree",
                        "Post grad": "Post grad",
                    },
                ),
                ui.input_slider("years", "Select Years of Coding:", 1, 50, 4),
                ui.input_select(
                    "dev",
                    "Select Dev Type",
                    selected="Data scientist or machine learning specialist",
                    choices={
                        "Developer, front-end": "Developer, front-end",
                        "Developer, full-stack": "Developer, full-stack",
                        "Developer, back-end": "Developer, back-end",
                        "Engineer, site reliability": "Engineer, site reliability",
                        "Senior Executive (C-Suite, VP, etc.)": "Senior Executive (C-Suite, VP, etc.)",
                        "Engineer, data": "Engineer, data",
                        "Developer, not-specified": "Developer, not-specified",
                        "Engineering manager": "Engineering manager",
                        "Project manager": "Project manager",
                        "Developer, desktop or enterprise applications": "Developer, desktop or enterprise applications",
                        "Data scientist or machine learning specialist": "Data scientist or machine learning specialist",
                        "Data or business analyst": "Data or business analyst",
                        "Security professional": "Security professional",
                        "Educator": "Educator",
                        "Academic researcher": "Academic researcher",
                        "Developer, QA or test": "Developer, QA or test",
                        "Database administrator": "Database administrator",
                        "DevOps specialist": "DevOps specialist",
                        "Product manager": "Product manager",
                        "Cloud infrastructure engineer": "Cloud infrastructure engineer",
                        "Developer, mobile": "Developer, mobile",
                        "Developer, game or graphics": "Developer, game or graphics",
                        "Developer, embedded applications or devices": "Developer, embedded applications or devices",
                        "Scientist": "Scientist",
                        "System administrator": "System administrator",
                        "Student": "Student",
                        "Designer": "Designer",
                        "Blockchain": "Blockchain",
                        "Marketing or sales professional": "Marketing or sales professional",
                    },
                ),
                ui.input_select(
                    "organization",
                    "Select Organization Type",
                    selected="10,000 or more employees",
                    choices={
                        "Just me - I am a freelancer, sole proprietor, etc.": "Just me - I am a freelancer, sole proprietor, etc.",
                        "2 to 9 employees": "2 to 9 employees",
                        "10 to 19 employees": "10 to 19 employees",
                        "20 to 99 employees": "20 to 99 employees",
                        "100 to 499 employees": "100 to 499 employees",
                        "500 to 999 employees": "500 to 999 employees",
                        "1,000 to 4,999 employees": "1,000 to 4,999 employees",
                        "5,000 to 9,999 employees": "5,000 to 9,999 employees",
                        "10,000 or more employees": "10,000 or more employees",
                    },
                ),
                ui.input_select(
                    "system",
                    "Select Operation System",
                    selected="MacOS",
                    choices={
                        "Windows": "Windows",
                        "MacOS": "MacOS",
                        "Linux-based": "Linux-based",
                        "Other": "Other",
                    },
                ),
                ui.input_select(
                    "age",
                    "Select your Age Group",
                    selected="25-34",
                    choices={
                        "Under 18": "Under 18",
                        "18-24": "18-24",
                        "25-34": "25-34",
                        "35-44": "35-44",
                        "45-54": "45-54",
                        "55-64": "55-64",
                        "65+": "65+",
                    },
                ),
                ui.tags.h4("Coders profile:"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_checkbox("APL", "APL"),
                        ui.input_checkbox("Ada", "Ada"),
                        ui.input_checkbox("Apex", "Apex"),
                        ui.input_checkbox("Assembly", "Assembly"),
                        ui.input_checkbox("Bash_Shell", "Bash/Shell"),
                        ui.input_checkbox("C", "C"),
                        ui.input_checkbox("CSharp", "C#"),
                        ui.input_checkbox("CPlusPlus", "C++"),
                        ui.input_checkbox("Clojure", "Clojure"),
                        ui.input_checkbox("Cobol", "Cobol"),
                        ui.input_checkbox("Crystal", "Crystal"),
                        ui.input_checkbox("Dart", "Dart"),
                        ui.input_checkbox("Delphi", "Delphi"),
                        ui.input_checkbox("Elixir", "Elixir"),
                        ui.input_checkbox("Erlang", "Erlang"),
                        ui.input_checkbox("FSharp", "FSharp"),
                        ui.input_checkbox("Flow", "Flow"),
                        ui.input_checkbox("Fortran", "Fortran"),
                        ui.input_checkbox("GDScript", "GDScript"),
                        ui.input_checkbox("Go", "Go"),
                        ui.input_checkbox("Groovy", "Groovy"),
                        ui.input_checkbox("HTML_CSS", "HTML/CSS"),
                        ui.input_checkbox("Haskell", "Haskell"),
                        ui.input_checkbox("Java", "Java"),
                        ui.input_checkbox("JavaScript", "JavaScript"),
                        ui.input_checkbox("Julia", "Julia"),
                    ),
                    ui.column(
                        6,
                        ui.input_checkbox("Kotlin", "Kotlin"),
                        ui.input_checkbox("Lisp", "Lisp"),
                        ui.input_checkbox("Lua", "Lua"),
                        ui.input_checkbox("Matlab", "Matlab"),
                        ui.input_checkbox("Nim", "Nim"),
                        ui.input_checkbox("OCaml", "OCaml"),
                        ui.input_checkbox("Objective_C", "Objective C"),
                        ui.input_checkbox("PHP", "PHP"),
                        ui.input_checkbox("Perl", "Perl"),
                        ui.input_checkbox("PowerShell", "PowerShell"),
                        ui.input_checkbox("Prolog", "Prolog"),
                        ui.input_checkbox("Python", "Python"),
                        ui.input_checkbox("R", "R"),
                        ui.input_checkbox("Raku", "Raku"),
                        ui.input_checkbox("Ruby", "Ruby"),
                        ui.input_checkbox("Rust", "Rust"),
                        ui.input_checkbox("SAS", "SAS"),
                        ui.input_checkbox("SQL", "SQL"),
                        ui.input_checkbox("Scala", "Scala"),
                        ui.input_checkbox("Solidity", "Solidity"),
                        ui.input_checkbox("Swift", "Swift"),
                        ui.input_checkbox("TypeScript", "TypeScript"),
                        ui.input_checkbox("VBA", "VBA"),
                        ui.input_checkbox("Visual_Basic_Net", "Visual Basic (.Net)"),
                        ui.input_checkbox("Zig", "Zig"),
                    ),
                ),
            ),
            ui.navset_tab(
                ui.nav_panel(
                    "Model prediction",
                    ui.tags.h2("Your main inputs:"),
                    ui.output_text_verbatim("country"),
                    ui.output_text_verbatim("education"),
                    ui.output_text_verbatim("years"),
                    ui.output_text_verbatim("dev"),
                    ui.output_text_verbatim("organization"),
                    ui.output_text_verbatim("system"),
                    ui.output_text_verbatim("age"),
                    ui.row(
                        ui.column(
                            4,
                            ui.output_text_verbatim("APL"),
                            ui.output_text_verbatim("Ada"),
                            ui.output_text_verbatim("Apex"),
                            ui.output_text_verbatim("Assembly"),
                            ui.output_text_verbatim("Bash_Shell"),
                            ui.output_text_verbatim("C"),
                            ui.output_text_verbatim("CSharp"),
                            ui.output_text_verbatim("CPlusPlus"),
                            ui.output_text_verbatim("Clojure"),
                            ui.output_text_verbatim("Cobol"),
                            ui.output_text_verbatim("Crystal"),
                            ui.output_text_verbatim("Dart"),
                            ui.output_text_verbatim("Delphi"),
                            ui.output_text_verbatim("Elixir"),
                            ui.output_text_verbatim("Erlang"),
                            ui.output_text_verbatim("FSharp"),
                            ui.output_text_verbatim("Flow"),
                            ui.output_text_verbatim("Fortran"),
                        ),
                        ui.column(
                            4,
                            ui.output_text_verbatim("GDScript"),
                            ui.output_text_verbatim("Go"),
                            ui.output_text_verbatim("Groovy"),
                            ui.output_text_verbatim("HTML_CSS"),
                            ui.output_text_verbatim("Haskell"),
                            ui.output_text_verbatim("Java"),
                            ui.output_text_verbatim("JavaScript"),
                            ui.output_text_verbatim("Julia"),
                            ui.output_text_verbatim("Kotlin"),
                            ui.output_text_verbatim("Lisp"),
                            ui.output_text_verbatim("Lua"),
                            ui.output_text_verbatim("Matlab"),
                            ui.output_text_verbatim("Nim"),
                            ui.output_text_verbatim("OCaml"),
                            ui.output_text_verbatim("Objective_C"),
                            ui.output_text_verbatim("PHP"),
                            ui.output_text_verbatim("Perl"),
                        ),
                        ui.column(
                            4,
                            ui.output_text_verbatim("PowerShell"),
                            ui.output_text_verbatim("Prolog"),
                            ui.output_text_verbatim("Python"),
                            ui.output_text_verbatim("R"),
                            ui.output_text_verbatim("Raku"),
                            ui.output_text_verbatim("Ruby"),
                            ui.output_text_verbatim("Rust"),
                            ui.output_text_verbatim("SAS"),
                            ui.output_text_verbatim("SQL"),
                            ui.output_text_verbatim("Scala"),
                            ui.output_text_verbatim("Solidity"),
                            ui.output_text_verbatim("Swift"),
                            ui.output_text_verbatim("TypeScript"),
                            ui.output_text_verbatim("VBA"),
                            ui.output_text_verbatim("Visual_Basic_Net"),
                            ui.output_text_verbatim("Zig"),
                        ),
                    ),
                    ui.div(style="height:25px;"),
                    ui.input_action_button(
                        "predict",
                        "Model Prediction",
                        icon=icon_svg("play"),
                        class_="btn-success",
                    ),
                    ui.div(style="height:25px;"),
                    ui.tags.h2("Output of the model will appear here:"),
                    ui.output_text_verbatim("xgb"),
                ),
                ui.nav_panel(
                    "Model description",
                    ui.tags.h2("About the model:"),
                    ui.tags.img(
                        src="https://www.researchgate.net/profile/Lara-Demajo/publication/350874464/figure/fig2/AS:1012594076827648@1618432649350/XGBoost-model-Source-Self.ppm",
                        height="100%",
                        width="100%",
                    ),
                    ui.tags.br(),
                    ui.tags.br(),
                    ui.tags.p(
                        "XGBoost, short for eXtreme Gradient Boosting, is an advanced and highly efficient implementation of gradient boosting, a machine learning technique used for regression, classification, and ranking problems. Developed as a project under the Distributed Machine Learning Community (DMLC), XGBoost has gained popularity for its performance and speed in machine learning competitions and real-world applications."
                    ),
                    ui.tags.p(
                        "Gradient Boosting Framework: At its core, XGBoost uses the gradient boosting framework, where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It's an ensemble technique that builds models sequentially, each new model correcting errors made by previous ones."
                    ),
                    ui.tags.p(
                        "Regularization: XGBoost includes L1 (Lasso Regression) and L2 (Ridge Regression) regularization terms in the objective function, which helps in reducing overfitting and improving model performance on unseen data."
                    ),
                    ui.tags.p(
                        "Handling of Sparse Data: XGBoost can efficiently handle sparse data (data with lots of zeros or missing values), making it suitable for a wide range of applications, including recommender systems and click prediction."
                    ),
                    ui.tags.p(
                        "Scalability and Efficiency: It is designed to be highly efficient, scalable, and portable. XGBoost can run on a single machine as well as on a distributed computing framework like Hadoop, and it utilizes resources optimally to achieve high performance."
                    ),
                    ui.tags.p(
                        "Flexibility: XGBoost allows users to define custom optimization objectives and evaluation criteria, adding a layer of flexibility that lets it be adapted to a wide range of domain-specific applications."
                    ),
                    ui.tags.p(
                        "Built-in Cross-validation: XGBoost has an integrated cross-validation tool at each iteration, making it easy to obtain accurate models without manually coding the cross-validation process."
                    ),
                    ui.tags.p(
                        "Handling of Missing Values: The algorithm has an in-built routine to handle missing values, allowing it to learn the best direction to take for missing values automatically."
                    ),
                    ui.download_button(
                        "downloadData",
                        "Download data.csv",
                        icon=icon_svg("download"),
                    ),
                    ui.div(style="height:10px;"),
                    ui.HTML(
                        '<p>Source data: <a href="https://insights.stackoverflow.com/survey" target="_blank">https://insights.stackoverflow.com/survey</a></p>'
                    ),
                ),
                ui.nav_panel(
                    "Salary research",
                    ui.tags.h2("Make model more accurate:"),
                    ui.tags.p(
                        "Fill your characteristics in the side bar menu, insert your yearly salary (USD) below and hit the 'Sent Data' button."
                    ),
                    ui.input_numeric(
                        "your_salary",
                        "Insert your yearly salary (USD)",
                        30000,
                        min=1,
                        max=1000000,
                        step=1,
                    ),
                    ui.input_action_button(
                        "sendSalary",
                        "Send Data",
                        icon=icon_svg("upload"),
                        class_="btn-success",
                    ),
                    # ui.tags.script(  # to reload after sending
                    #     """
                    #     document.getElementById('sendSalary').addEventListener('click', function() {
                    #         setTimeout(function() {
                    #             window.location.reload();
                    #         }, 7000); // 7000 milliseconds = 7 seconds
                    #     });
                    #     """
                    # ),
                    ui.div(style="height:10px;"),
                    ui.HTML(
                        "<p>After clicking just wait a few sec for a confirmation that will appear below...</p>"
                    ),
                    ui.output_ui("show_message"),
                ),
            ),
            border_radius=False,
            # border_color="black",
        ),
        # Footer
        ui.div(style="height:25px;"),
        ui.HTML(
            f"""
            <div style="text-align:center;">
                <p>Author's projects: <a href="https://jaroslavkotrba.com" style="text-decoration:none;" target="_blank">https://jaroslavkotrba.com</a></p>
                <a href="https://www.linkedin.com/in/jaroslav-kotrba/" target="_blank" style="font-size:24px;">{icon_svg("linkedin")}</a>
                <a href="https://github.com/JaroslavKotrba" target="_blank" style="font-size:24px;">{icon_svg("github")}</a>
                <a href="https://www.facebook.com/jaroslav.kotrba.9/" target="_blank" style="font-size:24px;">{icon_svg("facebook")}</a>
                <p>Copyright &copy; 2024</p>
            </div>
            """
        ),
    ),
    ui.nav_panel(
        "- PLOT -",
        ui.tags.h2("Visualisation of Salary Data (USD):"),
        ui.input_selectize(
            id="country_plot",
            label="Select Country",
            multiple=True,
            selected=[
                "Czech Republic",
                "Slovakia",
                "Poland",
                "Hungary",
                "Austria",
                "Germany",
                "France",
                "Switzerland",
                "UK",
                "USA",
            ],
            choices={
                "Argentina": "Argentina",
                "Australia": "Australia",
                "Austria": "Austria",
                "Bangladesh": "Bangladesh",
                "Belgium": "Belgium",
                "Brazil": "Brazil",
                "Bulgaria": "Bulgaria",
                "Canada": "Canada",
                "Chile": "Chile",
                "China": "China",
                "Colombia": "Colombia",
                "Croatia": "Croatia",
                "Czech Republic": "Czech Republic",
                "Denmark": "Denmark",
                "Estonia": "Estonia",
                "Finland": "Finland",
                "France": "France",
                "Germany": "Germany",
                "Greece": "Greece",
                "Hong Kong (S.A.R.)": "Hong Kong (S.A.R.)",
                "Hungary": "Hungary",
                "India": "India",
                "Iran": "Iran",
                "Ireland": "Ireland",
                "Israel": "Israel",
                "Italy": "Italy",
                "Japan": "Japan",
                "Lithuania": "Lithuania",
                "Malaysia": "Malaysia",
                "Mexico": "Mexico",
                "Netherlands": "Netherlands",
                "New Zealand": "New Zealand",
                "Norway": "Norway",
                "Pakistan": "Pakistan",
                "Poland": "Poland",
                "Portugal": "Portugal",
                "Romania": "Romania",
                "Russia": "Russia",
                "Serbia": "Serbia",
                "Singapore": "Singapore",
                "Slovakia": "Slovakia",
                "Slovenia": "Slovenia",
                "South Africa": "South Africa",
                "South Korea": "South Korea",
                "Spain": "Spain",
                "Sweden": "Sweden",
                "Switzerland": "Switzerland",
                "Taiwan": "Taiwan",
                "Turkey": "Turkey",
                "UK": "UK",
                "USA": "USA",
                "Ukraine": "Ukraine",
                "United Arab Emirates": "United Arab Emirates",
            },
        ),
        output_widget("plot1"),
        # Footer
        ui.div(style="height:10px;"),
        ui.HTML(
            f"""
            <div style="text-align:center;">
                <p>Author's projects: <a href="https://jaroslavkotrba.com" style="text-decoration:none;" target="_blank">https://jaroslavkotrba.com</a></p>
                <a href="https://www.linkedin.com/in/jaroslav-kotrba/" target="_blank" style="font-size:24px;">{icon_svg("linkedin")}</a>
                <a href="https://github.com/JaroslavKotrba" target="_blank" style="font-size:24px;">{icon_svg("github")}</a>
                <a href="https://www.facebook.com/jaroslav.kotrba.9/" target="_blank" style="font-size:24px;">{icon_svg("facebook")}</a>
                <p>Copyright &copy; 2024</p>
            </div>
            """
        ),
    ),
    ui.nav_panel(
        "- MAP -",
        ui.tags.h2("Interactive World Map (USD):"),
        output_widget("map1"),
        # Footer
        ui.div(style="height:10px;"),
        ui.HTML(
            f"""
            <div style="text-align:center;">
                <p>Author's projects: <a href="https://jaroslavkotrba.com" style="text-decoration:none;" target="_blank">https://jaroslavkotrba.com</a></p>
                <a href="https://www.linkedin.com/in/jaroslav-kotrba/" target="_blank" style="font-size:24px;">{icon_svg("linkedin")}</a>
                <a href="https://github.com/JaroslavKotrba" target="_blank" style="font-size:24px;">{icon_svg("github")}</a>
                <a href="https://www.facebook.com/jaroslav.kotrba.9/" target="_blank" style="font-size:24px;">{icon_svg("facebook")}</a>
                <p>Copyright &copy; 2024</p>
            </div>
            """
        ),
    ),
    # bg = "#158cba",
    # window_title = "SALARY of CODERs",
    underline=False,
    fillable_mobile=True,
    title=ui.tags.div(  # title must be at the end of the UI
        ui.tags.a(
            ui.tags.img(
                src="https://freeiconshop.com/wp-content/uploads/edd/wallet-outline-filled.png",
                style="height:50px;",
            ),
            href="https://jaroslavkotrba.shinyapps.io/salary_2023/",
        ),
        "SALARY of CODERs",
        style="display:flex; align-items:center; gap:10px;",
    ),
)


def server(input: Inputs, output: Outputs, session: Session):

    # - PREDICTION -

    # Model prediction section
    @output
    @render.text
    def country():
        return f'country: "{input.country()}"'

    @output
    @render.text
    def education():
        return f'education: "{input.education()}"'

    @output
    @render.text
    def years():
        return f'years of coding: "{input.years()}"'

    @output
    @render.text
    def dev():
        return f'dev type: "{input.dev()}"'

    @output
    @render.text
    def organization():
        return f'organization: "{input.organization()}"'

    @output
    @render.text
    def system():
        return f'system: "{input.system()}"'

    @output
    @render.text
    def age():
        return f'age: "{input.age()}"'

    @output
    @render.text
    def APL():
        return f'APL: "{str(int(input.APL()))}"'

    @output
    @render.text
    def Ada():
        return f'Ada: "{str(int(input.Ada()))}"'

    @output
    @render.text
    def Apex():
        return f'Apex: "{str(int(input.Apex()))}"'

    @output
    @render.text
    def Bash_Shell():
        return f'Bash/Shell: "{str(int(input.Bash_Shell()))}"'

    @output
    @render.text
    def C():
        return f'C: "{str(int(input.C()))}"'

    @output
    @render.text
    def CSharp():
        return f'C#: "{str(int(input.CSharp()))}"'

    @output
    @render.text
    def CPlusPlus():
        return f'C++: "{str(int(input.CPlusPlus()))}"'

    @output
    @render.text
    def Clojure():
        return f'Clojure: "{str(int(input.Clojure()))}"'

    @output
    @render.text
    def Cobol():
        return f'Cobol: "{str(int(input.Cobol()))}"'

    @output
    @render.text
    def Crystal():
        return f'Crystal: "{str(int(input.Crystal()))}"'

    @output
    @render.text
    def Dart():
        return f'Dart: "{str(int(input.Dart()))}"'

    @output
    @render.text
    def Delphi():
        return f'Delphi: "{str(int(input.Delphi()))}"'

    @output
    @render.text
    def Elixir():
        return f'Elixir: "{str(int(input.Elixir()))}"'

    @output
    @render.text
    def Erlang():
        return f'Erlang: "{str(int(input.Erlang()))}"'

    @output
    @render.text
    def FSharp():
        return f'F#: "{str(int(input.FSharp()))}"'

    @output
    @render.text
    def Flow():
        return f'Flow: "{str(int(input.Flow()))}"'

    @output
    @render.text
    def Fortran():
        return f'Fortran: "{str(int(input.Fortran()))}"'

    @output
    @render.text
    def GDScript():
        return f'GDScript: "{str(int(input.GDScript()))}"'

    @output
    @render.text
    def Go():
        return f'Go: "{str(int(input.Go()))}"'

    @output
    @render.text
    def Groovy():
        return f'Groovy: "{str(int(input.Groovy()))}"'

    @output
    @render.text
    def HTML_CSS():
        return f'HTML/CSS: "{str(int(input.HTML_CSS()))}"'

    @output
    @render.text
    def Haskell():
        return f'Haskell: "{str(int(input.Haskell()))}"'

    @output
    @render.text
    def Java():
        return f'Java: "{str(int(input.Java()))}"'

    @output
    @render.text
    def JavaScript():
        return f'JavaScript: "{str(int(input.JavaScript()))}"'

    @output
    @render.text
    def Julia():
        return f'Julia: "{str(int(input.Julia()))}"'

    @output
    @render.text
    def Kotlin():
        return f'Kotlin: "{str(int(input.Kotlin()))}"'

    @output
    @render.text
    def Lisp():
        return f'Lisp: "{str(int(input.Lisp()))}"'

    @output
    @render.text
    def Lua():
        return f'Lua: "{str(int(input.Lua()))}"'

    @output
    @render.text
    def Matlab():
        return f'Matlab: "{str(int(input.Matlab()))}"'

    @output
    @render.text
    def Nim():
        return f'Nim: "{str(int(input.Nim()))}"'

    @output
    @render.text
    def OCaml():
        return f'OCaml: "{str(int(input.OCaml()))}"'

    @output
    @render.text
    def Objective_C():
        return f'Objective-C: "{str(int(input.Objective_C()))}"'

    @output
    @render.text
    def PHP():
        return f'PHP: "{str(int(input.PHP()))}"'

    @output
    @render.text
    def Perl():
        return f'Perl: "{str(int(input.Perl()))}"'

    @output
    @render.text
    def PowerShell():
        return f'PowerShell: "{str(int(input.PowerShell()))}"'

    @output
    @render.text
    def Prolog():
        return f'Prolog: "{str(int(input.Prolog()))}"'

    @output
    @render.text
    def Python():
        return f'Python: "{str(int(input.Python()))}"'

    @output
    @render.text
    def R():
        return f'R: "{str(int(input.R()))}"'

    @output
    @render.text
    def Raku():
        return f'Raku: "{str(int(input.Raku()))}"'

    @output
    @render.text
    def Ruby():
        return f'Ruby: "{str(int(input.Ruby()))}"'

    @output
    @render.text
    def Rust():
        return f'Rust: "{str(int(input.Rust()))}"'

    @output
    @render.text
    def SAS():
        return f'SAS: "{str(int(input.SAS()))}"'

    @output
    @render.text
    def SQL():
        return f'SQL: "{str(int(input.SQL()))}"'

    @output
    @render.text
    def Scala():
        return f'Scala: "{str(int(input.Scala()))}"'

    @output
    @render.text
    def Solidity():
        return f'Solidity: "{str(int(input.Solidity()))}"'

    @output
    @render.text
    def Swift():
        return f'Swift: "{str(int(input.Swift()))}"'

    @output
    @render.text
    def TypeScript():
        return f'TypeScript: "{str(int(input.TypeScript()))}"'

    @output
    @render.text
    def VBA():
        return f'VBA: "{str(int(input.VBA()))}"'

    @output
    @render.text
    def Visual_Basic_Net():
        return f'Visual Basic (.Net): "{str(int(input.Visual_Basic_Net()))}"'

    @output
    @render.text
    def Zig():
        return f'Zig: "{str(int(input.Zig()))}"'

    @reactive.Effect  # predict button - init
    @reactive.event(
        input.predict
    )  # predict button - calculation will happen after the button click for the first time
    def _():
        # Model pipeline
        def print_regression_metrics(model):
            # Import
            import numpy as np
            import pandas as pd

            df = pd.read_csv("data/survey_clean.csv")
            df

            # Splitting
            X = df.drop(columns=["Salary"])
            X
            y = df.Salary
            y

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0
            )

            # Preprocessing
            from sklearn.compose import make_column_transformer
            from sklearn.preprocessing import OneHotEncoder

            column_trans = make_column_transformer(
                (
                    OneHotEncoder(),
                    [
                        "Country",
                        "EdLevel",
                        "DevType",
                        "OrgSize",
                        "OrgSize",
                        "OpSys",
                        "Age",
                    ],
                ),  # non-numeric
                remainder="passthrough",
            )

            # Scaling
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            # Making pipeline
            from sklearn.pipeline import make_pipeline

            pipe = make_pipeline(column_trans, scaler, model)
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            outcome = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
            outcome["difference"] = outcome["y_test"] - outcome["y_pred"]
            outcome["difference_percentage"] = round(
                outcome.difference / (outcome.y_test / 100), 6
            )

            model_name = str(model).split("(")[0]
            if "Regression" in model_name:
                model_name = model_name.replace("Regression", "")

            print(f"{model_name} regression model:")
            print("MAPE: ", round(outcome.difference_percentage.abs().mean(), 2), "%")
            print("MAE: ", round(mean_absolute_error(y_test, y_pred), 4))
            print("RMSE: ", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
            print("R2:", round(r2_score(y_test, y_pred), 4))

            return [pipe, model_name, outcome, y_test, y_pred]

        # Model sample
        def sample(pipe):
            # Import
            import numpy as np
            import pandas as pd

            # X_sample
            X_sample = np.array(
                [
                    input.country(),
                    input.education(),
                    input.years(),
                    input.dev(),
                    input.organization(),
                    input.system(),
                    input.age(),
                    str(int(input.APL())),
                    str(int(input.Ada())),
                    str(int(input.Apex())),
                    str(int(input.Assembly())),
                    str(int(input.Bash_Shell())),
                    str(int(input.C())),
                    str(int(input.CSharp())),
                    str(int(input.CPlusPlus())),
                    str(int(input.Clojure())),
                    str(int(input.Cobol())),
                    str(int(input.Crystal())),
                    str(int(input.Dart())),
                    str(int(input.Delphi())),
                    str(int(input.Elixir())),
                    str(int(input.Erlang())),
                    str(int(input.FSharp())),
                    str(int(input.Flow())),
                    str(int(input.Fortran())),
                    str(int(input.GDScript())),
                    str(int(input.Go())),
                    str(int(input.Groovy())),
                    str(int(input.HTML_CSS())),
                    str(int(input.Haskell())),
                    str(int(input.Java())),
                    str(int(input.JavaScript())),
                    str(int(input.Julia())),
                    str(int(input.Kotlin())),
                    str(int(input.Lisp())),
                    str(int(input.Lua())),
                    str(int(input.Matlab())),
                    str(int(input.Nim())),
                    str(int(input.OCaml())),
                    str(int(input.Objective_C())),
                    str(int(input.PHP())),
                    str(int(input.Perl())),
                    str(int(input.PowerShell())),
                    str(int(input.Prolog())),
                    str(int(input.Python())),
                    str(int(input.R())),
                    str(int(input.Raku())),
                    str(int(input.Ruby())),
                    str(int(input.Rust())),
                    str(int(input.SAS())),
                    str(int(input.SQL())),
                    str(int(input.Scala())),
                    str(int(input.Solidity())),
                    str(int(input.Swift())),
                    str(int(input.TypeScript())),
                    str(int(input.VBA())),
                    str(int(input.Visual_Basic_Net())),
                    str(int(input.Zig())),
                ]
            )

            X_sample = pd.DataFrame(X_sample.reshape(1, -1))
            X_sample.columns = [
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

            # Pipe
            y_pred = pipe.predict(X_sample)
            y_pred = np.where(y_pred < 0, 0, y_pred)  # to be not negative

            return y_pred

        # XGB Regression - model calculation
        @output
        @render.text
        def xgb():
            import xgboost as xgb

            model = xgb.XGBRegressor(
                objective="reg:squarederror",  # Objective function to be used
                n_estimators=128,  # Number of gradient boosted trees
                max_depth=6,  # Maximum tree depth for base learners
                learning_rate=0.3,  # Boosting learning rate
                gamma=0,  # Minimum loss reduction required to make a further partition on a leaf node - prevent overfitting
                min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
                subsample=1,  # Subsample ratio of the training instance
                colsample_bytree=1,  # Subsample ratio of columns when constructing each tree
                colsample_bylevel=1,  # Subsample ratio of columns for each level
                colsample_bynode=1,  # Subsample ratio of columns for each split
                reg_alpha=1,  # L1 regularization term on weights
                reg_lambda=1,  # L2 regularization term on weights
                scale_pos_weight=1,  # Balancing of positive and negative weights
                base_score=0.5,  # The initial prediction score of all instances, global bias
                booster="gbtree",  # Which booster to use: gbtree, gblinear or dart
                random_state=0,  # Random number seed for reproducibility
            )

            output = print_regression_metrics(model)

            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            pipe = output[0]
            model_name = output[1]
            outcome = output[2]
            y_test = output[3]
            y_pred = output[4]

            with reactive.isolate():  # predict button - isolate this event
                return f"{model_name} model: \nYour salary yearly: {round(float(sample(pipe)[0]),2)} USD \nYour salary yearly: {round(float(sample(pipe)[0]) * 0.9241,2)} EUR \nYour salary monthly: {round(float(sample(pipe)[0]) * 22.210 / 12,2)} CZK \n\nMAPE: {round(outcome.difference_percentage.abs().mean(),2)}% \nMAE: {round(mean_absolute_error(y_test, y_pred),2)} \nRMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred)),2)} \nR2: {round(r2_score(y_test, y_pred),2)}"

    # Model description section
    @render.download(filename="data.csv")
    def downloadData():
        yield df.to_string(index=False)

    # Model research section
    message = reactive.Value("")

    @output
    @render.ui  # Use render.ui to allow HTML rendering
    def show_message():
        if message():
            return ui.HTML(message())

    async def hide_message_after_delay():
        await asyncio.sleep(1)
        message.set("")

    @reactive.Effect  # send button - init
    @reactive.event(
        input.sendSalary
    )  # predict button - calculation will happen after the button click for the first time
    def _():
        def sendSalary():
            import gspread

            gc = gspread.service_account(filename="credentials/google-sheets-api.json")

            sh = gc.open("Salary_2023-Shiny-Python")
            worksheet = sh.sheet1

            user = [
                input.your_salary(),
                input.country(),
                input.education(),
                input.years(),
                input.dev(),
                input.organization(),
                input.system(),
                input.age(),
                int(input.APL()),
                int(input.Ada()),
                int(input.Apex()),
                int(input.Assembly()),
                int(input.Bash_Shell()),
                int(input.C()),
                int(input.CSharp()),
                int(input.CPlusPlus()),
                int(input.Clojure()),
                int(input.Cobol()),
                int(input.Crystal()),
                int(input.Dart()),
                int(input.Delphi()),
                int(input.Elixir()),
                int(input.Erlang()),
                int(input.FSharp()),
                int(input.Flow()),
                int(input.Fortran()),
                int(input.GDScript()),
                int(input.Go()),
                int(input.Groovy()),
                int(input.HTML_CSS()),
                int(input.Haskell()),
                int(input.Java()),
                int(input.JavaScript()),
                int(input.Julia()),
                int(input.Kotlin()),
                int(input.Lisp()),
                int(input.Lua()),
                int(input.Matlab()),
                int(input.Nim()),
                int(input.OCaml()),
                int(input.Objective_C()),
                int(input.PHP()),
                int(input.Perl()),
                int(input.PowerShell()),
                int(input.Prolog()),
                int(input.Python()),
                int(input.R()),
                int(input.Raku()),
                int(input.Ruby()),
                int(input.Rust()),
                int(input.SAS()),
                int(input.SQL()),
                int(input.Scala()),
                int(input.Solidity()),
                int(input.Swift()),
                int(input.TypeScript()),
                int(input.VBA()),
                int(input.Visual_Basic_Net()),
                int(input.Zig()),
            ]

            worksheet.append_row(user)
            print("Google sheet successfully updated!")

            session.send_input_message("your_salary", {"value": ""})

            message.set(
                "<div style='padding: 10px; border: 2px solid green; border-radius: 5px; background-color: #e6ffe6; color: green;'>"
                "Data successfully updated!"
                "</div>"
            )
            asyncio.create_task(hide_message_after_delay())

        with reactive.isolate():  # predict button - isolate this event
            sendSalary()

    # - PLOT -

    import numpy as np
    import pandas as pd
    import plotly.express as px

    df = pd.read_csv("data/survey_clean.csv")
    df

    @reactive.Calc
    def country_filtered():
        return df[df["Country"].isin(input.country_plot())]

    @render_widget
    def plot1():
        df_filtered = country_filtered()

        plot1 = px.box(
            df_filtered.sort_values("Country"),
            x="Country",
            y="Salary",
            template="simple_white",
        )
        plot1.update_traces(marker_color="#158cba")
        plot1.update_xaxes(
            tickangle=90, tickmode="array", tickvals=df_filtered["Country"].unique()
        )

        return plot1

    # - MAP -

    salary_by_country = df.groupby("Country")["Salary"].median().reset_index()

    import plotly.express as px

    map1 = px.choropleth(
        salary_by_country,
        locations="Country",
        locationmode="country names",
        color="Salary",
        color_continuous_scale=px.colors.diverging.RdBu,
        labels={"Salary": "Median Salary"},
    )

    map1.update_layout(
        geo=dict(
            showframe=True,
            showcoastlines=True,
            showocean=True,
            oceancolor="LightBlue",
            projection_type="equirectangular",
        ),
        coloraxis_showscale=False,
    )

    register_widget("map1", map1)


app = App(app_ui, server)
