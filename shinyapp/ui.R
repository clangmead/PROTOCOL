#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#
library(shiny)
library(shinydashboard)
library(shinyFeedback)
library(sortable)
library(DT)

sideBar <- dashboardSidebar(
  sidebarMenu(
    id = "tabs",
    menuItem("Start page", tabName = "start", icon = icon("start")),
    menuItem("Experiments upload", tabName = "exp", icon=icon("exp_input"))
    # menuItem("Optimization history", tabName = "opt_hist", icon = icon("opt_hist"))
  )
)


body <- dashboardBody(
  useShinyFeedback(),
  tabItems(
    tabItem(tabName = "start",
            fluidRow(box(h3("Job type"),
                         selectInput(
                           "analysis_type",
                           "Initial/additional round",
                           c(Initial = "initial",Additional = "additional"),
                           selected = "initial"
                         )
            ),
            box(
              conditionalPanel(condition = "input.analysis_type == 'initial'",
                               h3("Parameters setup"),
                               numericInput("param_num","No. of parameters",value = 6, min=1L, max = 10L)
                               
                               
              )))
            ,
                                  
            wellPanel(
              uiOutput("params_input")
            ),
            
            conditionalPanel(condition = "input.analysis_type == 'additional'",
                             wellPanel(
                               h3("Previous object"),
                               fileInput("bayesOptObj",
                                         "Upload previous BayesOpt result",
                                         multiple = FALSE)
                             )
            ),
            
            actionButton("param_confirm","Confirm Parameters")
            
    ),
    
    
    tabItem(tabName = "exp",
            selectInput("prev_data","Do you want to upload previous experiments?",c("Yes","No")),
            conditionalPanel("input.prev_data == 'Yes'",
                             fileInput("prev_exps",
                                       "Upload previous experiments",
                                       multiple = FALSE),
                             actionButton("exp_upload_confirm"," Upload")
                             ),
            conditionalPanel("input.prev_data == 'No'",
                             p("You only need to change this value if you want to include algorithms other than PROTOCOL"),
                             numericInput(inputId = "no_sgst_param",
                                          label = "No. of initial parameter sets", 
                                          value = 1,min = 1, max = 10, step = 1),
                             actionButton("suggest_param"," Get suggested parameters")),
            
            uiOutput("sgt_exps"),
            dataTableOutput("sgt_exps_tbl"),
            uiOutput("exp_hist"),
            dataTableOutput("exp_hist_tbl"),
            h3("Optimization setup"),
            fluidRow(
              # column(4,
              #        radioButtons("acq_type","Batch Selection method",
              #                     choices = c("Expected Improvement" = "ei",
              #                                 "Upper Confidence Bound"="ucb",
              #                                 "Possibility of Improvement"= "poi",
              #                                 "Thompson Sampling"= "TS"), 
              #                     selected = "EI")
              #        ),
              column(4,
                     checkboxGroupInput("acq_type_multi", "Batch Selection Methods",
                                        choices = c("PROTOCOL" = "ptl",
                                                    "Expected Improvement" = "ei",
                                                    "Upper Confidence Bound"="ucb",
                                                    "Possibility of Improvement"= "poi",
                                                    "Thompson Sampling"= "TS"),
                                        selected = "EI")
                     ),
              column(4,
                     radioButtons("opt_mode","Mode of optimization", choices = c("Sequential"=1,"Batch"=2), selected = 2),
                     conditionalPanel("input.opt_mode == 2",
                                      numericInput("batch_size","Batch size",value = 3))
              ),
              column(4,
                     actionButton("run_opt","Run optimization"),
                     actionButton("test_PROTOCOL","test protocol")
                     )
            ),
            DTOutput("new_params_tbl"),
            conditionalPanel("is.null(output.new_params_tbl) == FALSE",
                             actionButton("update", "Update experiments"))
            
    ),
    
    tabItem(tabName = "opt_hist",
            downloadButton("downloadData","Download"),
            dataTableOutput("all_exps"))
  )
  
)# End of Dashboard body

# fluidPage(
dashboardPage(
  skin = "purple",
  dashboardHeader(title = "PROTOCOL"),
  sideBar,
  body
)
