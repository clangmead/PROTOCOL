#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#
# setwd("D:/20Fall/Thesis/shiny/Bayes_opt")
library(shiny)
library(shinydashboard)
library(readxl)
library(zoo)
library(rBayesianOptimization)
library(data.table)
library(magrittr)
library(GPfit)
library(rlist)
library(kernlab)
library(foreach)
library(dplyr)
library(reticulate)
library(shinyjqui)
source("utils.R")
reticulate::use_condaenv("ml",required = TRUE)
source_python("../PROTOCOL.py")


shinyServer(function(input, output,session) {
  
  output$params_input <- renderUI({
    numParams <- input$param_num
    # Create a panel for each parameter
    # maldi_params <- c("Acc","Grid_Pct","Delay","Shots_per_spectrum")
    hplc_params <- c("Sol_ratio","Grad","Flow_rate","Inj_col","Col_temp","Abs_wvl")
    lapply(1:numParams, function(i){
      box( title = paste0("Param ",i), width = 3, solidHeader = T, 
           textInput(paste0("param_name_",i),paste0("Param ",i," label"),hplc_params[i]),
        radioButtons(paste0("param_type_",i),paste0("Param ",i," type"),
                     choices = c("discrete"=1,"continous"=2),
                     selected = 1
                     ),
        conditionalPanel(condition = paste0("input.param_type_",i,"==1"),
                         #if discrete, set labels
                         radioButtons(paste0("discrete_type_",i), paste0("Ordinal or Nominal"),
                                      choices = c("Ordinal"=1, "Nominal"=2),
                                      selected = 1),
                         #If ordinal
                         conditionalPanel(condition = paste0("input.discrete_type_",i,"==1"),
                                          textInput(paste0("ordinal_lbls_",i),"Labels of levels")
                                          ,
                                          conditionalPanel(
                                            condition = paste0("input.ordinal_lbls_",i),
                                            orderInput(paste0("order_",i),paste0("Order the labels"),items = NULL, width = '300px')
                                                             
                                          )
                                          ),
                         #If Noninal
                         conditionalPanel(condition = paste0("input.discrete_type_",i,"==2"),
                                          textInput("categorical_lvls", "Levels of the parameters (separated by commma)")
                                          )
          
        ),
        #if continuous, set lower and upper bound
        conditionalPanel(condition = paste0("input.param_type_",i,"==2"),
                          wellPanel(
                            numericInput(paste0("lb_",i),paste0("Lower bound of the ",i," th param"),
                                         value = 0.0),
                            numericInput(paste0("ub_",i),paste0("Upper bound of the ",i," th param"),
                                         value = 10.0)

                         ))
      )
    })
  })
  
  num_params <- reactive({
    input$param_num
  })
  
  # params_names <- reactive({
  #   params <- c()
  #   for (i in 1:num_params()) {
  #     params_names <- c(params_names, eval(parse(text = paste0("input$param_name_",i))))
  #   }
  #   params
  # })
  

  ordinal_lbls <- reactive({
    req(num_params())
    req(input$ordinal_lbls_1)

    res <- lapply(1:num_params(), function(i){
      string <- eval(parse(text = paste0("input$ordinal_lbls_",i)))
      unlist(strsplit(string, ","))
    })
    res
  })
  
  # Update order input
  observe({
    ordinal_lbls()
    lapply(1:num_params(), 
           function(i){
             observeEvent(paste0(input$ordinal_lbls_,"i"),{
               updateOrderInput(session,
                                paste0("order_",i),
                                items = unlist(strsplit(ordinal_lbls()[[i]],",")))

               })
           })
  })
  

  # Get the length of ordinal labels
  ordinal_lengths <- reactive({
    req(ordinal_lbls())
    res <- lapply(1:num_params(), function(i){
      length(unlist(strsplit(ordinal_lbls()[[i]],",")))
    })
    res
  })
  
  
  #----------------------- Generate search bounds -----------------
  searchbound <- reactive({
    search_bound <- as.data.frame(matrix(nrow = num_params(),ncol = 3))
    for (i in 1:num_params()) {
      #Continuous
      if(eval(parse(text = paste0("input$param_type_",i,"==2")))){
        # Set lower bound
        search_bound[i,1] <- eval(parse(text = paste0("input$lb_",i)))
        search_bound[i,2] <- eval(parse(text = paste0("input$ub_",i)))
        search_bound[i,3] <- "numeric"
      }else{
        #Ordinal
        if(eval(parse(text = paste0("input$discrete_type_",i,"==1")))){
          search_bound[i,1] <- 1L
          search_bound[i,2] <- ordinal_lengths()[[i]]
          search_bound[i,3] <- "integer"
        }
      }     
    }
    search_bound
  })
  
  #--------------------------------------------------------------------
  
  
  #------------------------ When parameters are confirmed --------------
  
  DT_bounds <- reactiveVal()
  
  observeEvent(input$param_confirm,{
    search_bound <- searchbound()
    params_names <- c()
    
    for (i in 1:num_params()) {
      params_names <- c(params_names, eval(parse(text = paste0("input$param_name_",i))))
    }
    

    DT_bounds_df <- cbind(params_names, search_bound)
    colnames(DT_bounds_df) <- c("Parameter","Lower","Upper","Type")
    DT_bounds(as.data.table(DT_bounds_df))
    print(DT_bounds())
    
    updateTabItems(session, "tabs", "exp")
    
  })


  #------------------ Generate the table containing all evaluated parameters ------------
  DT_hist_show <- reactiveVal()
  
  observeEvent(input$exp_upload_confirm,{
    
    exp_hist_file <- input$prev_exps$datapath
    DT_history <- read.delim(exp_hist_file, header = FALSE, sep = "\t")
    num_params = input$param_num
    
    params_names <- c()
    for (i in 1:num_params()) {
      params_names <- c(params_names, eval(parse(text = paste0("input$param_name_",i))))
    }
    
    col_names <- c(params_names,"objective")
    colnames(DT_history) <- col_names
    DT_hist_show(DT_history)
  })
  

  
  #-------------------------------------------------
  # Backend history DF
  DT_hist <- reactive({
    DT_hist <- DT_hist_show()
    # Encode ordinal params
    for(i in 1: num_params()){
      if(eval(parse(text = paste0("input$discrete_type_",i,"==1")))){
        DT_hist[,i] <- encode_ord_param(DT_hist[,i],eval(parse(text = paste0("input$order_",i))))
      }
    }
    print(DT_hist)
    
    DT_hist
  })
  
  #------------------------ Upload previous experiments ------------------
  toListen <- reactive({
    list(input$prev_exps,input$exp_upload_confirm)
  })
  
  
  observeEvent(toListen(),{
    if (!is.na(input$prev_exps) && input$exp_upload_confirm == 1){
      output$exp_hist <- renderUI(
        h3("Historical experiments")
      )
      output$exp_hist_tbl <- renderDataTable(DT_hist_show())
    }

  })
  
  observe({input$prev_data
    input$update
    if(input$prev_data == "No" && input$update == 1){
      output$exp_hist <- renderUI(
        h3("Historical experiments")
      )
      output$exp_hist_tbl <- renderDataTable(DT_hist_show())
    }
  }

  )
  
  observeEvent(input$suggest_param,{
    
    params_names <- c()
    for (i in 1:num_params()) {
      params_names <- c(params_names, eval(parse(text = paste0("input$param_name_",i))))
    }
    
    output$sgt_exps <- renderUI(
      h3(paste0(input$no_sgst_param," suggested initial parameters"))
    )
    suggested_df <- as.data.frame(matrix(nrow = 0, ncol = num_params()))
    for(i in c(1:input$no_sgst_param)){
      suggested_df <- rbind(suggested_df, extract_one_set_of_params(DT_bounds()))
    }
    colnames(suggested_df) <- params_names
    # output$sgt_exps_tbl <- renderDataTable(suggested_df)
    output$exp_hist_tbl <- renderDataTable(suggested_df)
    
  })
  #----------------------------------------------------
  
  #-------------------- When starting optimization ---------------------
  
  new_par <- reactiveVal()
  
  ##############################
  observeEvent(input$test_PROTOCOL,{
    params_names <- c()
    for (i in 1:num_params()) {
      params_names <- c(params_names, eval(parse(text = paste0("input$param_name_",i))))
    }
    
    logging_dir <- "test_log/"
    continous <- FALSE
    batch_size <- 3
    max_evals <- 25
    xmin <- np_array(DT_bounds()[,Lower])
    xmax <- np_array(DT_bounds()[,Upper])
    par_types <- DT_bounds()[,Type]
    num_decimals <- ifelse(par_types == "integer",0L,2L)
    data <- DT_hist()
    d1 <- as.data.frame(initialize(logging_dir, continous, batch_size, max_evals,
                     xmin = xmin, xmax = xmax,
                     num_decimals = num_decimals))
    colnames(d1) <- params_names
    
    # Encode ordinal parameters
    for(i in c(1:num_params())){
      if(eval(parse(text = paste0("input$param_type_",i,"==1")))&eval(parse(text = paste0("input$discrete_type_",i,"==1")))){
        d1[,i] <- decode_ord_param(d1[,i],eval(parse(text = paste0("input$order_",i))))
      }
    }
    # Add objective column
    df <- cbind(d1, objective=rep(0, nrow(d1)))
    print(df)
    new_par(df)
  })
  ##############################
  
  observeEvent(input$run_opt,{
    # print(input$acq_type_multi)
    # req(DT_hist(), input$opt_mode, input$acq_type_multi)
    # validate(need(nrow(DT_hist()<3), "Please upload historical experiments or get suggested initial experiments!"))
    # Get all selected acquisition funcs
    acqs <- reactive({
      acq_selected <- !is.null(input$acq_type_multi)
      feedbackDanger("acq_type_multi", !acq_selected, "Please selected acquisition function(s)")
      req(acq_selected, cancelOutput = TRUE)
      input$acq_type_multi
      
    })
    req(nrow(DT_hist()>1), input$opt_mode)
    q <- input$batch_size
    if (input$opt_mode == 1){# Sequential mode
      print("sequential")

      new_params <- opt_one_round_sequential(DT_bounds(), DT_hist(),acqs)
    }else if(input$opt_mode == 2){# Batch mode
      print("hi")

      new_params <- opt_one_round_batch_multi_acq(DT_bounds(),DT_hist(),q,1.3,acqs)
    }
    print(new_params)
    
    # Encode ordinal parameters
    for(i in c(1:num_params())){
      if(eval(parse(text = paste0("input$param_type_",i,"==1")))&eval(parse(text = paste0("input$discrete_type_",i,"==1")))){
        new_params[,i] <- decode_ord_param(new_params[,i],eval(parse(text = paste0("input$order_",i))))
      }
    }
    # Add objective column
    df <- cbind(new_params, objective=rep(0, nrow(new_params)))
    new_par(df)
  })
  
  # new parameters selected by the user
  selected_new_pars <- reactive({
    curr_new_pars <- new_par()
    curr_new_pars[input$new_params_tbl_rows_selected,]
  }

  )
  
  # Detect edits to the new param DF
  observeEvent(input$new_params_tbl_cell_edit,{
    new_exps <- new_par()

    cell <- input$new_params_tbl_cell_edit
    new_exps[cell$row, cell$col] <- cell$value
    new_par(new_exps)
    })
  
  
  observeEvent(input$run_opt,
               output$new_params_tbl <-  renderDT(new_par(), editable="cell")
               
               )
  
  observeEvent(input$test_PROTOCOL,
               output$new_params_tbl <-  renderDT(new_par(), editable="cell")
               
  )
  
  # Update evaluated experiments
  observeEvent(input$update,{
    DT_history <- DT_hist_show()
    new_recs <- selected_new_pars()
    # update front-end df
    if(!is.null(DT_history)){
      DT_hist_show(rbind(DT_history,new_recs[,-(ncol(new_recs)-1)]))
      
    }else{
      DT_hist_show(new_recs)
    }
  })
  #--------------------------------------------------------------
  
  output$downloadData <- downloadHandler(
    filename = function(){
      "OPT_results.txt"
    },
    content = function(file){
      write.table(DT_hist_show(),file, quote = FALSE, row.names = FALSE)
    }
  )
  })
  

